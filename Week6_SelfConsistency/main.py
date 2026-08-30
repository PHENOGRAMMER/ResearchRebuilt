# main.py
from sampler import sample_paths
from aggregator import majority_vote, greedy_answer, normalize
from colorama import Fore, Style, init
import time

init()

# Curated test cases — designed to show where self-consistency helps:
#  1. Intuitive-trap questions (model's System-1 gets it wrong, diversity helps)
#  2. Multi-step arithmetic (different reasoning chains may make different errors)
#  3. Questions where greedy is already strong (SC should match with high confidence)
TEST_CASES = [
    {
        "question": "A bat and ball together cost $1.10. The bat costs exactly $1.00 more than the ball. How much does the ball cost in dollars?",
        "expected": "0.05",
        "why": "CRT problem — greedy path often takes the intuitive wrong answer 0.10"
    },
    {
        "question": "A train travels 60 km in the first hour, 80 km in the second hour, and 100 km in the third hour. What is the average speed in km/h over the entire trip?",
        "expected": "80",
        "why": "Multi-step — total distance 240 / total time 3 = 80; models sometimes average the speeds instead"
    },
    {
        "question": "In a transformer with 12 attention heads, each with a key dimension of 64, what is the total dimension of the query matrix Q if all heads are concatenated?",
        "expected": "768",
        "why": "ML architecture arithmetic — 12 * 64 = 768"
    },
    {
        "question": "You fine-tune a model on 10,000 examples for 3 epochs with batch size 32. Assume the last incomplete batch in each epoch is still processed as one step. How many gradient update steps does training take?",
        "expected": "939",
        "why": "ceil(10000/32) * 3 = 313 * 3 = 939 — rounding catches most models"
    },
    {
        "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "expected": "9",
        "why": "Classic trick question — 'all but 9' means 9 remain, models often compute 17-9=8"
    },
    {
        "question": "If you have a 7-minute hourglass and an 11-minute hourglass, what is the shortest time you can measure exactly 15 minutes?",
        "expected": "15",
        "why": "Lateral reasoning — start both, when 7 runs out flip it, when 11 runs out flip 7 again (4 min left) = 11+4=15"
    },
    {
        "question": "What is 37 * 43?",
        "expected": "1591",
        "why": "Harder multiplication — models often make carry errors; SC corrects via diverse attempts"
    },
]

N_SAMPLES = 5  # number of reasoning paths — paper uses 5-40

def check_correct(answer: str, expected: str) -> bool:
    """Check if answer matches expected using the same normalization as voting."""
    norm_answer = normalize(answer)
    norm_expected = normalize(expected)
    return norm_answer == norm_expected

print(f"\n{Fore.CYAN}{'='*65}")
print(f"Self-Consistency — Test-Time Scaling via Width")
print(f"Model: qwen/qwen3.8-27b | N={N_SAMPLES} paths per question")
print(f"{'='*65}{Style.RESET_ALL}\n")

summary = []

for tc in TEST_CASES:
    print(f"\n{Fore.CYAN}── {tc['question'][:80]}{Style.RESET_ALL}")
    print(f"   Why: {tc['why']}")
    print(f"   Expected: {tc['expected']}\n")

    # Sample N reasoning paths
    print(f"  Sampling {N_SAMPLES} paths...")
    paths = sample_paths(tc['question'], n=N_SAMPLES, temperature=0.8)

    # Greedy baseline — first path only
    greedy = greedy_answer(paths)
    greedy_ok = check_correct(greedy, tc['expected'])

    # Self-consistency — majority vote over all paths
    result = majority_vote(paths)
    sc_ok = check_correct(result['winner'], tc['expected'])

    # Print individual paths
    print(f"\n  {Fore.YELLOW}Individual reasoning paths:{Style.RESET_ALL}")
    for p in paths:
        mark = f"{Fore.GREEN}✓" if check_correct(p['answer'], tc['expected']) \
               else f"{Fore.RED}✗"
        print(f"  Path {p['path_id']}: {mark} {p['answer'][:60]}{Style.RESET_ALL}")

    # Print vote distribution
    print(f"\n  {Fore.YELLOW}Vote distribution:{Style.RESET_ALL}")
    for norm_ans, info in result['distribution'].items():
        bar = "█" * info['count']
        winner_tag = " ← WINNER" if norm_ans == result['winner_normalized'] else ""
        print(f"  {bar} ({info['count']}/{N_SAMPLES}) {info['example'][:50]}{winner_tag}")

    # Print comparison
    greedy_color = Fore.GREEN if greedy_ok else Fore.RED
    sc_color = Fore.GREEN if sc_ok else Fore.RED
    print(f"\n  {greedy_color}[Greedy]          {'✓' if greedy_ok else '✗'} {greedy[:70]}{Style.RESET_ALL}")
    print(f"  {sc_color}[Self-Consistency] {'✓' if sc_ok else '✗'} {result['winner'][:70]} "
          f"({result['votes']}/{result['total']} votes, {result['confidence']:.0%} confidence){Style.RESET_ALL}")

    total_tokens = sum(p['tokens'] for p in paths)
    summary.append({
        "q": tc['question'][:45],
        "expected": tc['expected'],
        "greedy_ok": greedy_ok,
        "sc_ok": sc_ok,
        "confidence": result['confidence'],
        "tokens": total_tokens,
        "votes": f"{result['votes']}/{N_SAMPLES}"
    })

# Summary table
print(f"\n{Fore.CYAN}{'='*65}")
print("RESULTS SUMMARY")
print(f"{'='*65}{Style.RESET_ALL}")
print(f"{'Question':<47} {'Greedy':>7} {'SC':>4} {'Conf':>6} {'Votes':>7} {'Tokens':>8}")
print("-" * 80)

greedy_total = sc_total = 0
for s in summary:
    g = f"{Fore.GREEN}✓{Style.RESET_ALL}" if s['greedy_ok'] else f"{Fore.RED}✗{Style.RESET_ALL}"
    sc = f"{Fore.GREEN}✓{Style.RESET_ALL}" if s['sc_ok'] else f"{Fore.RED}✗{Style.RESET_ALL}"
    greedy_total += s['greedy_ok']
    sc_total += s['sc_ok']
    print(f"{s['q']:<47} {g:>7} {sc:>4} {s['confidence']:>5.0%} {s['votes']:>7} {s['tokens']:>8}")

print("-" * 80)
print(f"{'Total Correct':<47} {greedy_total}/{len(summary)}        {sc_total}/{len(summary)}")
print(f"\n{Fore.YELLOW}Paper finding: Self-consistency consistently outperforms greedy decoding")
print(f"by sampling diverse paths and marginalizing via majority vote.")
print(f"No training. No verifier. Just sample more.{Style.RESET_ALL}")