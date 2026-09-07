import re

# main.py
from difficulty_classifier import classify_difficulty
from strategies import greedy, best_of_n, sequential_refinement, compute_optimal
from colorama import Fore, Style, init

init()

TEST_CASES = [
    {
        "question": "What is 12 * 15?",
        "expected": "180",
        "label": "EASY — simple multiplication"
    },
    {
        "question": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "expected": "0.05",
        "label": "MEDIUM — CRT algebra"
    },
    {
        "question": """You have a 3-gallon jug and a 5-gallon jug.
Using only these two jugs and unlimited water, measure exactly 4 gallons.
Give the exact step-by-step sequence.""",
        "expected": "fill the 5",
        "label": "MEDIUM-HARD — multi-step planning"
    },
    {
        "question": """A snail is at the bottom of a 10-foot well.
Each day it climbs 3 feet. Each night it slides back 2 feet.
On which day does the snail reach the top?""",
        "expected": "8",
        "label": "HARD — classic trap (answer is day 8, not day 10)"
    },
    {
        "question": """In a RAG pipeline using FAISS, you have 1M documents each with 768-dimensional embeddings stored as float32.
How much RAM (in GB) does the index require? Show your calculation.""",
        "expected": "2.88",
        "label": "HARD — ML systems arithmetic (1M * 768 * 4 bytes / 1B)"
    },
]

BUDGET = "medium"

def check(answer: str, expected: str) -> bool:
    actual = answer.lower().replace('$', '').replace(',', '')
    expected_text = expected.lower().replace('$', '').replace(',', '')

    expected_numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', expected_text)
    if expected_numbers:
        actual_numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', actual)
        return all(number in actual_numbers for number in expected_numbers)

    stop_words = {"a", "an", "the", "is", "are", "of", "to", "and"}
    expected_words = {
        word for word in re.findall(r"[a-z]+", expected_text)
        if word not in stop_words
    }
    return expected_words.issubset(set(re.findall(r"[a-z]+", actual)))

print(f"\n{Fore.CYAN}{'='*65}")
print("Compute-Optimal Test-Time Scaling Recreation")
print(f"Paper: Snell et al. 2024 | Budget: {BUDGET}")
print(f"{'='*65}{Style.RESET_ALL}\n")

summary = []

for tc in TEST_CASES:
    print(f"\n{Fore.CYAN}── [{tc['label']}]{Style.RESET_ALL}")
    print(f"   Q: {tc['question'][:100].strip()}")
    print(f"   Expected: {tc['expected']}\n")

    # Step 1: Classify difficulty
    diff = classify_difficulty(tc['question'])
    diff_color = {
        "easy": Fore.GREEN, "medium": Fore.YELLOW, "hard": Fore.RED
    }.get(diff['difficulty'], Fore.WHITE)

    print(f"  {diff_color}[Difficulty Classifier] → {diff['difficulty'].upper()} "
          f"(confidence: {diff['confidence']:.0%}){Style.RESET_ALL}")
    print(f"   Reason: {diff['reason']}")

    # Step 2: Greedy baseline
    g = greedy(tc['question'])
    g_ok = check(g['answer'], tc['expected'])
    g_color = Fore.GREEN if g_ok else Fore.RED
    print(f"\n  {g_color}[Greedy]  {'✓' if g_ok else '✗'} {g['answer'][:80]} "
          f"| {g['tokens']} tokens{Style.RESET_ALL}")

    # Step 3: Compute-optimal routing
    co = compute_optimal(tc['question'], diff['difficulty'], BUDGET)
    co_ok = check(co['answer'], tc['expected'])
    co_color = Fore.GREEN if co_ok else Fore.RED
    print(f"  {co_color}[Compute-Optimal] {'✓' if co_ok else '✗'} {co['answer'][:80]} "
          f"| {co['tokens']} tokens{Style.RESET_ALL}")
    print(f"   Routed to: {co['strategy']}")

    # Show refinement history for sequential
    if co.get('routed_to') == 'sequential' and 'history' in co:
        print(f"   Refinement trace:")
        for h in co['history']:
            print(f"     Round {h['round']}: {h['answer'][:60]}")

    # Show vote distribution for Best-of-N
    if co.get('all_answers'):
        from collections import Counter
        votes = Counter(a[:40].lower() for a in co['all_answers'])
        print(f"   Vote distribution: {dict(votes.most_common(3))}")

    improvement = co['tokens'] / max(g['tokens'], 1)
    summary.append({
        "label": tc['label'][:35],
        "difficulty": diff['difficulty'],
        "greedy_ok": g_ok,
        "co_ok": co_ok,
        "greedy_tokens": g['tokens'],
        "co_tokens": co['tokens'],
        "strategy": co.get('routed_to', ''),
        "compute_ratio": improvement
    })

# Summary table
print(f"\n{Fore.CYAN}{'='*65}")
print("RESULTS SUMMARY — Greedy vs Compute-Optimal")
print(f"{'='*65}{Style.RESET_ALL}")
print(f"{'Question':<37} {'Diff':>6} {'Greedy':>7} {'CO':>4} {'Strategy':>12} {'Ratio':>7}")
print("-" * 80)

g_total = co_total = 0
for s in summary:
    g = f"{Fore.GREEN}✓{Style.RESET_ALL}" if s['greedy_ok'] else f"{Fore.RED}✗{Style.RESET_ALL}"
    co = f"{Fore.GREEN}✓{Style.RESET_ALL}" if s['co_ok'] else f"{Fore.RED}✗{Style.RESET_ALL}"
    g_total += s['greedy_ok']
    co_total += s['co_ok']
    d_color = {
        "easy": Fore.GREEN, "medium": Fore.YELLOW, "hard": Fore.RED
    }.get(s['difficulty'], Fore.WHITE)
    diff_str = f"{d_color}{s['difficulty']:<6}{Style.RESET_ALL}"
    print(f"{s['label']:<37} {diff_str} {g:>7} {co:>4} "
          f"{s['strategy']:>12} {s['compute_ratio']:>6.1f}x")

print("-" * 80)
print(f"{'Total Correct':<37}        {g_total}/{len(summary)}      {co_total}/{len(summary)}")
print(f"\n{Fore.YELLOW}Paper finding: Difficulty-adaptive routing outperforms fixed strategies.")
print(f"Easy problems: Best-of-N. Hard problems: Sequential refinement.")
print(f"Compute-optimal small model can beat a 14x larger greedy model.{Style.RESET_ALL}")