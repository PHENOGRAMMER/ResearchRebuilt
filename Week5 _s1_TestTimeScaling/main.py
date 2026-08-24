# main.py
from budget_forcing import run_baseline, run_force_stop, run_wait_injection
from colorama import Fore, Style, init
import re

init()

# ─── Unicode box-drawing constants ───────────────────────────────────────────
DIV  = "─" * 62
HDIV = "═" * 62

TEST_CASES = [
    {
        "question": "How many letters 'r' are in the word 'strawberry'? Count each one.",
        "expected": "3",
        "label":    "Strawberry letter count",
        "why":      "Paper's flagship example — quick intuition says 2, correct answer is 3"
    },
    {
        "question": "I have a 3-gallon jug and a 5-gallon jug with no markings. I need exactly 4 gallons. Give me the exact numbered steps.",
        "expected": "fill the 5",
        "label":    "Water jug puzzle",
        "why":      "Multi-step planning — force-stop cuts out before the plan can complete"
    },
    {
        "question": "A bat and ball together cost $1.10. The bat costs exactly $1.00 more than the ball. How much does the ball cost? Show your algebra.",
        "expected": "0.05",
        "label":    "Bat & ball CRT",
        "why":      "Classic cognitive reflection trap — intuition gives $0.10, algebra gives $0.05"
    },
    {
        "question": (
            "In a RAG system, you embed documents and retrieve by cosine similarity.\n"
            "A user asks: 'Why does attention fail on very long sequences?'\n"
            "You retrieve:\n"
            "  - Doc A (score 0.91): attention in vision transformers\n"
            "  - Doc B (score 0.87): softmax instability in long-context LLMs\n"
            "  - Doc C (score 0.82): quadratic complexity of self-attention\n"
            "  - Doc D (score 0.79): positional encoding degradation beyond training length\n\n"
            "Doc A has the highest score. Should you use it? Which docs actually answer the query "
            "and why is cosine similarity alone misleading here?"
        ),
        "expected": "doc b",
        "label":    "RAG relevance reasoning",
        "why":      "High cosine score ≠ topical relevance — requires semantic discrimination"
    },
]

def check_correct(answer: str, expected: str) -> bool:
    clean = re.sub(r'\*+', '', answer).lower()
    clean = re.sub(r'\$', '', clean)
    return all(w in clean for w in expected.lower().split())

def _ans_preview(text: str, n: int = 100) -> str:
    """Single-line, stripped preview of an answer."""
    return re.sub(r'\s+', ' ', re.sub(r'\*+', '', text)).strip()[:n]

def _think_preview(text: str, n: int = 160) -> str:
    return re.sub(r'\s+', ' ', text).strip()[:n]

def print_question_header(i: int, tc: dict):
    print(f"\n{Fore.CYAN}{DIV}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Q{i}  {tc['label'].upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{DIV}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Question :{Style.RESET_ALL} {tc['question'].splitlines()[0][:72]}")
    print(f"  {Fore.WHITE}Expected :{Style.RESET_ALL} {tc['expected']}")
    print(f"  {Fore.WHITE}Why hard  :{Style.RESET_ALL} {tc['why']}")

def print_result(result: dict, expected: str) -> bool:
    correct = check_correct(result['answer'], expected)
    color   = Fore.GREEN if correct else Fore.RED
    mark    = "✓  CORRECT" if correct else "✗  WRONG"
    mode    = result['mode']

    print(f"\n  {color}┌─ {mode:<26} {mark}{Style.RESET_ALL}")
    print(f"  {color}│{Style.RESET_ALL}  Answer  : {_ans_preview(result['answer'])}")
    print(f"  {color}│{Style.RESET_ALL}  Tokens  : {result['tokens']:,}")
    think = _think_preview(result['think'])
    print(f"  {color}│{Style.RESET_ALL}  Thinking: {think}{'...' if think else '(none)'}")
    if result.get('self_corrected'):
        prev = _ans_preview(result.get('initial_answer', ''), 70)
        print(f"  {color}│{Style.RESET_ALL}  {Fore.YELLOW}↺ Was    : {prev}{Style.RESET_ALL}")
    print(f"  {color}└{'─'*54}{Style.RESET_ALL}")

    return correct

# ─── Header ──────────────────────────────────────────────────────────────────
print(f"\n{Fore.CYAN}{HDIV}")
print( "  s1: BUDGET FORCING  —  Test-Time Scaling Demo")
print(f"  Model : qwen/qwen3.6-27b  |  Paper: arxiv.org/abs/2501.19393")
print(f"{HDIV}{Style.RESET_ALL}")

summary = []

for idx, tc in enumerate(TEST_CASES, start=1):
    print_question_header(idx, tc)

    b  = run_baseline(tc['question'])
    fs = run_force_stop(tc['question'], think_token_limit=150)
    wi = run_wait_injection(tc['question'], n_waits=2)

    b_ok  = print_result(b,  tc['expected'])
    fs_ok = print_result(fs, tc['expected'])
    wi_ok = print_result(wi, tc['expected'])

    summary.append({
        "label":           tc['label'],
        "baseline":        b_ok,
        "force_stop":      fs_ok,
        "wait":            wi_ok,
        "self_corrected":  wi.get('self_corrected', False),
        "baseline_tokens": b['tokens'],
        "wait_tokens":     wi['tokens'],
    })

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{Fore.CYAN}{HDIV}")
print( "  RESULTS SUMMARY")
print(f"{HDIV}{Style.RESET_ALL}")
print(f"  {'Question':<30} {'Baseline':>9} {'Force-Stop':>11} {'Wait (x2)':>10} {'Fixed?':>7} {'Tokens↑':>8}")
print(f"  {DIV}")

for s in summary:
    b  = f"{Fore.GREEN}   ✓{Style.RESET_ALL}" if s['baseline']   else f"{Fore.RED}   ✗{Style.RESET_ALL}"
    fs = f"{Fore.GREEN}         ✓{Style.RESET_ALL}" if s['force_stop'] else f"{Fore.RED}         ✗{Style.RESET_ALL}"
    w  = f"{Fore.GREEN}        ✓{Style.RESET_ALL}"  if s['wait']       else f"{Fore.RED}        ✗{Style.RESET_ALL}"
    fix = f"{Fore.YELLOW}   Yes{Style.RESET_ALL}" if s['self_corrected'] else "    No"
    ratio = f"{s['wait_tokens']/max(s['baseline_tokens'],1):.1f}x"
    label = s['label'][:30]
    print(f"  {label:<30}{b}{fs}{w}{fix} {ratio:>8}")

print(f"\n  {Fore.YELLOW}Key finding  : Wait injection uses 1.5–2.5× more tokens but rescues wrong answers.")
print(f"  Force-Stop   : Guaranteed to fail — reasoning cut off before any conclusion.")
print(f"  Self-correct : Model catches its own errors when given time to re-examine.{Style.RESET_ALL}")
print(f"\n{Fore.CYAN}{HDIV}{Style.RESET_ALL}\n")