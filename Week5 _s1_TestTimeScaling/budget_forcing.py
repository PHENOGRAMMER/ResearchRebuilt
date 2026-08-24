# budget_forcing.py
from groq import Groq
from colorama import Fore, Style
import os, re
import time
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "qwen/qwen3.6-27b"

THINK_PROMPT = "You are a careful reasoning assistant. Think through problems step by step inside <think> tags, then give your final answer outside the tags."
NO_THINK_PROMPT = "You are a direct answer assistant. Do NOT think, do NOT use <think> tags, do NOT explain. Give only the final answer in one plain sentence."

def call_groq_with_retry(*args, **kwargs):
    for attempt in range(10):
        try:
            return client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                print(f"  {Fore.YELLOW}[rate limit — waiting 15s]{Style.RESET_ALL}")
                time.sleep(15)
            else:
                raise
    raise Exception("Max retries exceeded for Groq API")

def clean_markdown(text: str) -> str:
    """Strip markdown bold/italic/dollar signs for clean display."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\$', '', text)
    return text.strip()

def extract_parts(text: str) -> tuple[str, str]:
    """Extract <think> content and final answer — strip all XML tags from answer."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    # Strip the think block from text to isolate answer
    answer_section = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    answer_clean = re.sub(r'<[^>]+>', '', answer_section).strip()

    # Fallback: last 300 chars of full text minus think block
    if not answer_clean:
        clean_full = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        clean_full = re.sub(r'<[^>]+>', '', clean_full)
        answer_clean = clean_full.strip()[-300:]

    return think, answer_clean

def run_baseline(question: str) -> dict:
    """Standard inference — no budget intervention."""
    resp = call_groq_with_retry(
        model=MODEL,
        messages=[
            {"role": "system", "content": THINK_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.6,
        max_tokens=2048
    )
    text = resp.choices[0].message.content
    think, answer = extract_parts(text)
    return {
        "mode": "Baseline",
        "think": think,
        "answer": answer,
        "tokens": resp.usage.completion_tokens,
        "raw": text
    }

def run_force_stop(question: str, think_token_limit: int = 150) -> dict:
    """
    Force-stop: truncate thinking budget, then demand a final answer.
    Paper's 'budget forcing' with a low token ceiling.
    """
    # Step 1: Hard token cap — cuts thinking mid-stream
    resp = call_groq_with_retry(
        model=MODEL,
        messages=[
            {"role": "system", "content": THINK_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.6,
        max_tokens=think_token_limit
    )
    truncated = resp.choices[0].message.content
    think_preview = re.sub(r'<[^>]+>', '', truncated).strip()[:200]

    # Step 2: Force a clean answer using a NO-THINK system prompt
    force_resp = call_groq_with_retry(
        model=MODEL,
        messages=[
            {"role": "system", "content": NO_THINK_PROMPT},
            {"role": "user", "content": f"Based on partial reasoning below, give your best final answer in one sentence.\n\nPartial reasoning: {think_preview[:300]}\n\nQuestion: {question}"}
        ],
        temperature=0.1,
        max_tokens=80
    )
    raw_answer = force_resp.choices[0].message.content.strip()
    # Strip any tags that leaked through
    clean_answer = re.sub(r'<[^>]+>', '', raw_answer).strip()

    return {
        "mode": "Force-Stop",
        "think": think_preview,
        "answer": clean_answer,
        "tokens": resp.usage.completion_tokens + force_resp.usage.completion_tokens,
        "raw": truncated
    }

def run_wait_injection(question: str, n_waits: int = 2) -> dict:
    """
    Wait injection: the paper's core mechanism.
    Suppresses end-of-thinking, injects 'Wait' to force self-correction.
    """
    # Step 1: Initial generation
    resp1 = call_groq_with_retry(
        model=MODEL,
        messages=[
            {"role": "system", "content": THINK_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.6,
        max_tokens=1024
    )
    initial_output = resp1.choices[0].message.content
    initial_think, initial_answer = extract_parts(initial_output)
    total_tokens = resp1.usage.completion_tokens

    current_think = initial_think
    current_answer = initial_answer

    # Step 2: Inject Wait N times — force re-examination
    for i in range(n_waits):
        if '</think>' in initial_output:
            think_so_far = initial_output.split('</think>')[0].replace('<think>', '').strip()
        else:
            think_so_far = initial_output

        wait_prompt = f"""<think>
{think_so_far}
Wait, let me re-examine this carefully. I may have made an error above. Let me verify each step from scratch:
"""
        wait_resp = call_groq_with_retry(
            model=MODEL,
            messages=[
                {"role": "system", "content": THINK_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": wait_prompt}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        wait_output = wait_resp.choices[0].message.content
        total_tokens += wait_resp.usage.completion_tokens

        new_think, new_answer = extract_parts(wait_output)
        if new_answer:
            current_think = new_think
            current_answer = new_answer
        initial_output = wait_output

    return {
        "mode": f"Wait-Injection (x{n_waits})",
        "think": current_think,
        "initial_answer": initial_answer,
        "answer": current_answer,
        "tokens": total_tokens,
        "self_corrected": initial_answer.strip() != current_answer.strip() and bool(current_answer)
    }