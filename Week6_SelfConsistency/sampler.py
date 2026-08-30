# sampler.py
from groq import Groq
import os, re
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "qwen/qwen3.8-27b"

COT_PROMPT = """Solve this step by step. Show your full reasoning.
On the very last line, write ONLY: Answer: <number>
Give a single numeric answer (no units, no words, just the number)."""

def extract_answer(text: str) -> str:
    """Extract final answer from reasoning path."""
    # Try 'Answer:' prefix first
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try </think> block — take text after it
    if '</think>' in text:
        after = text.split('</think>', 1)[1].strip()
        clean = re.sub(r'<[^>]+>', '', after).strip()
        # Try to find an Answer: line in the post-think text
        match2 = re.search(r'Answer:\s*(.+?)(?:\n|$)', clean, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        # Take first non-empty line
        for line in clean.split('\n'):
            if line.strip():
                return line.strip()[:120]

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1][:120] if lines else ""


def sample_paths(question: str, n: int = 5, temperature: float = 0.8) -> list[dict]:
    """
    Sample N diverse reasoning paths.
    Temperature > 0 is critical — diversity comes from stochastic sampling.
    Paper uses temperature 0.5–1.0 depending on the model.
    """
    paths = []
    for i in range(n):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"{COT_PROMPT}\n\nQuestion: {question}"
            }],
            temperature=temperature,
            max_tokens=1024
        )
        text = resp.choices[0].message.content
        answer = extract_answer(text)
        paths.append({
            "path_id": i + 1,
            "reasoning": text[:300],  # preview only
            "answer": answer,
            "tokens": resp.usage.completion_tokens
        })
    return paths