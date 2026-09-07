# strategies.py
from groq import Groq
from collections import Counter
import os, re
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

def extract_answer(text: str) -> str:
    """Extract the final answer while ignoring reasoning and formatting debris."""
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    if '</think>' in clean:
        clean = clean.split('</think>', 1)[1]
    clean = re.sub(r'<[^>]+>', '', clean).strip()

    labeled = re.findall(
        r'(?:final answer|answer|therefore|thus)\s*[:\-]?\s*(.+)',
        clean,
        flags=re.IGNORECASE,
    )
    if labeled:
        return _clean_candidate(labeled[-1])

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    for line in reversed(lines):
        if not re.fullmatch(r'[|\\`*_\-\s]+', line):
            return _clean_candidate(line)
    return _clean_candidate(clean)


def _clean_candidate(candidate: str) -> str:
    """Remove markdown/table wrappers that make equivalent answers vote differently."""
    candidate = candidate.strip().strip('`*_')
    candidate = re.sub(r'^\s*[|\\]+\s*|\s*[|\\]+\s*$', '', candidate)
    candidate = re.sub(r'\s+', ' ', candidate)
    return candidate[:200]


def _vote_key(answer: str) -> str:
    """Prefer the numeric/conclusion-bearing part when comparing sampled answers."""
    normalized = answer.lower().replace(',', '')
    numbers = re.findall(r'(?<![a-z])[-+]?\d+(?:\.\d+)?', normalized)
    if numbers:
        return '|'.join(numbers[-2:])
    words = re.findall(r'[a-z]+', normalized)
    return ' '.join(words[-12:])


def _final_prompt(question: str) -> str:
    return (
        f"Solve this problem carefully:\n{question}\n\n"
        "Reason privately, then finish with exactly one line in this format: "
        "FINAL ANSWER: <short answer>. Do not put a table or markdown on that line."
    )


def greedy(question: str) -> dict:
    """Baseline: single greedy path, no scaling."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _final_prompt(question)}],
        temperature=0.0,
        max_tokens=512
    )
    text = resp.choices[0].message.content
    return {
        "strategy": "Greedy (No Scaling)",
        "answer": extract_answer(text),
        "tokens": resp.usage.completion_tokens,
        "paths": 1
    }


def best_of_n(question: str, n: int = 5) -> dict:
    """
    Parallel Best-of-N: sample N independent paths, majority vote.
    Paper: effective for easy-medium problems and high compute budgets.
    Equivalent to self-consistency (Week 6) — wide, parallel.
    """
    answers = []
    total_tokens = 0

    for _ in range(n):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _final_prompt(question)}],
            temperature=0.8,
            max_tokens=512
        )
        text = resp.choices[0].message.content
        answers.append(extract_answer(text))
        total_tokens += resp.usage.completion_tokens

    # Majority vote (no PRM — we use vote count as proxy for reward)
    normalized = [_vote_key(a) for a in answers]
    most_common = Counter(normalized).most_common(1)[0]
    winner_idx = normalized.index(most_common[0])

    return {
        "strategy": f"Best-of-N (N={n}, parallel)",
        "answer": answers[winner_idx],
        "tokens": total_tokens,
        "paths": n,
        "votes": f"{most_common[1]}/{n}",
        "all_answers": answers
    }


def sequential_refinement(question: str, rounds: int = 3) -> dict:
    """
    Sequential beam-style refinement: each round critiques and improves previous answer.
    Paper: most effective for hard problems at low-medium budgets.
    This is the sequential / depth-first strategy — narrow but iterative.
    """
    total_tokens = 0
    current_answer = ""
    history = []

    # Round 1: initial attempt
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _final_prompt(question)}],
        temperature=0.6,
        max_tokens=600
    )
    current_text = resp.choices[0].message.content
    current_answer = extract_answer(current_text)
    total_tokens += resp.usage.completion_tokens
    history.append({"round": 1, "answer": current_answer})

    # Rounds 2+: critique and refine
    for r in range(2, rounds + 1):
        refine_resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": _final_prompt(question)},
                {"role": "assistant", "content": current_text},
                {"role": "user", "content": """Review your solution carefully:
1. Check every step for arithmetic or logical errors
2. Verify your answer satisfies all constraints in the problem
3. If you find an error, correct it. If the solution is correct, confirm it.
Provide your final answer clearly using exactly one final line: FINAL ANSWER: <short answer>."""}
            ],
            temperature=0.3,
            max_tokens=600
        )
        current_text = refine_resp.choices[0].message.content
        current_answer = extract_answer(current_text)
        total_tokens += refine_resp.usage.completion_tokens
        history.append({"round": r, "answer": current_answer})

    return {
        "strategy": f"Sequential Refinement ({rounds} rounds)",
        "answer": current_answer,
        "tokens": total_tokens,
        "paths": rounds,
        "history": history
    }


def compute_optimal(question: str, difficulty: str, budget: str = "medium") -> dict:
    """
    Compute-optimal routing — the paper's core contribution.
    Routes to the right strategy based on difficulty + budget.

    Paper finding:
    - Easy + any budget → Best-of-N (parallel, low compute)
    - Hard + low budget → Sequential refinement (beam-style, focused)
    - Hard + high budget → Best-of-N with more samples
    - Medium + medium budget → Best-of-N moderate N
    """
    routing = {
        ("easy",   "low"):    ("best_of_n",   2),
        ("easy",   "medium"): ("best_of_n",   3),
        ("easy",   "high"):   ("best_of_n",   5),
        ("medium", "low"):    ("sequential",  2),
        ("medium", "medium"): ("best_of_n",   4),
        ("medium", "high"):   ("best_of_n",   6),
        ("hard",   "low"):    ("sequential",  3),
        ("hard",   "medium"): ("sequential",  4),
        ("hard",   "high"):   ("best_of_n",   8),
    }

    strategy_type, n = routing.get((difficulty, budget), ("best_of_n", 4))

    if strategy_type == "sequential":
        result = sequential_refinement(question, rounds=n)
    else:
        result = best_of_n(question, n=n)

    result["strategy"] = f"Compute-Optimal → {result['strategy']} [difficulty={difficulty}, budget={budget}]"
    result["routed_to"] = strategy_type
    return result