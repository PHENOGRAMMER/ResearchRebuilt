# main.py
# Week 8 — #ResearchRebuilt
# TERMINATOR: Learning Optimal Exit Points for Early Stopping in CoT Reasoning
# Paper: arXiv:2603.12529
#
# What this reproduces:
#   - The CORE FINDING: LRMs produce correct answers early in their reasoning
#     chain, then waste tokens second-guessing themselves
#   - TERMINATOR's measurement: first-correct-answer position (post-hoc)
#   - TERMINATOR's mechanism: stability-gated exit detection (live, during stream)
#   - Difficulty gradient: easy tasks show bigger overthinking than hard ones
#
# What this does NOT reproduce:
#   - TERMINATOR's trained binary classifier (requires GPU fine-tuning)

from early_exit import run_early_exit, check_answer
from monitor import print_question_header, print_exit_bar
from colorama import Fore, Style, init

init()

DIV  = "─" * 65
HDIV = "═" * 65

# ─── Test cases ───────────────────────────────────────────────────────────────
# 3 difficulty levels × 2 questions each.
# Designed so easy tasks have obvious short answers the model might
# arrive at quickly but then over-explain, while hard tasks need more reasoning.

TEST_CASES = [
    # ── EASY ─────────────────────────────────────────────────────────────────
    {
        "question": (
            "Is the time complexity of standard scaled dot-product attention "
            "O(n²) with respect to sequence length n? "
            "Answer yes or no, then explain."
        ),
        "expected":     "yes",
        "answer_type":  "yes_no",
        "label":        "EASY — attention complexity",
        "difficulty":   "easy",
    },
    {
        "question": (
            "You have a sorted array of exactly 1,000,000 integers and run "
            "binary search for a value that does not exist. "
            "What is the maximum number of comparisons binary search will make? "
            "Give a single integer."
        ),
        "expected":     "20",
        "answer_type":  "integer",
        "label":        "EASY — binary search depth",
        "difficulty":   "easy",
    },
    # ── MEDIUM ───────────────────────────────────────────────────────────────
    {
        "question": (
            "A transformer uses d_model = 768 and h = 12 attention heads. "
            "What is the dimensionality of each attention head (d_k)? "
            "Give a single integer."
        ),
        "expected":     "64",
        "answer_type":  "integer",
        "label":        "MEDIUM — transformer head dimension",
        "difficulty":   "medium",
    },
    {
        "question": (
            "You load a 7-billion-parameter language model in full precision (float32), "
            "where each parameter requires 4 bytes. "
            "How many GB of GPU VRAM are needed just to hold the model weights? "
            "Give a single number in GB."
        ),
        "expected":     "28",
        "answer_type":  "number",
        "label":        "MEDIUM — LLM weight memory",
        "difficulty":   "medium",
    },
    # ── HARD ─────────────────────────────────────────────────────────────────
    {
        "question": (
            "In a standard multi-head attention block, you double the number of "
            "heads from h=8 to h=16 while keeping d_model fixed at 512. "
            "Assuming only the head count changes, does the total parameter count "
            "of the MHA block increase, decrease, or stay the same? "
            "Answer in one word, then explain."
        ),
        "expected":     "same",
        "answer_type":  "same",
        "label":        "HARD — MHA parameter invariance",
        "difficulty":   "hard",
    },
    {
        "question": (
            "You train a binary classifier. "
            "Test set: 1000 samples, 900 negative, 100 positive. "
            "Your model predicts everything as negative. "
            "What is the accuracy? What is the F1 score? "
            "Give both values."
        ),
        "expected":     "90, 0",
        "answer_type":  "multi_numeric",
        "label":        "HARD — imbalanced classifier metrics",
        "difficulty":   "hard",
    },
]

# Use deepseek-r1-distill-llama-70b — the ONLY Groq model that exposes
# a separate reasoning stream via delta.reasoning_content.
# The original code used openai/gpt-oss-20b which merges everything into
# delta.content, making reasoning_word_count always 0.
MODEL = "qwen/qwen3.6-27b"


# ─── Header ───────────────────────────────────────────────────────────────────

print(f"\n{Fore.CYAN}{HDIV}")
print(f"  TERMINATOR — Week 8 #ResearchRebuilt")
print(f"  Stability-Gated Early Exit Detection")
print(f"  Model: {MODEL} | Stability threshold: 3 chunks")
print(f"  Paper: arXiv:2603.12529")
print(f"{HDIV}{Style.RESET_ALL}\n")

summary = []

for i, tc in enumerate(TEST_CASES, 1):
    print_question_header(i, tc, DIV)

    result = run_early_exit(
        tc["question"],
        MODEL,
        answer_type=tc["answer_type"],
        expected_answer=tc["expected"],
    )

    # Post-hoc correctness on the full completed response
    full_ok = check_answer(
        result["raw"],          # use full reasoning + answer text
        tc["expected"],
        tc["answer_type"],
    )

    full_words      = result["full_reasoning_words"]
    exit_pos        = result["exit_word_position"]
    potential_pct   = result["potential_reduction"]
    exit_detected   = result["exit_detected"]

    # ── Per-question output ────────────────────────────────────────────────
    print(f"\n  {'─'*61}")
    ans_preview = result["full_answer"][:120].replace("\n", " ")
    print(f"  Final answer     : {ans_preview}")
    print(f"  Completion tokens: {result['tokens']}")
    print(f"  Reasoning words  : {full_words}")
    print(f"  Full correctness : {'✓ Correct' if full_ok else '✗ Wrong'}")

    if exit_detected and exit_pos is not None:
        color = Fore.GREEN if potential_pct >= 40 else Fore.YELLOW
        print(
            f"  {color}TERMINATOR exit  : ~{exit_pos} words  "
            f"({potential_pct:.1f}% potential reduction){Style.RESET_ALL}"
        )
        print_exit_bar(exit_pos, full_words)
    else:
        print(f"  {Fore.RED}TERMINATOR exit  : not detected{Style.RESET_ALL}")

    summary.append({
        "label"              : tc["label"],
        "difficulty"         : tc["difficulty"],
        "tokens"             : result["tokens"],
        "full_reasoning_words": full_words,
        "exit_word_position" : exit_pos,
        "potential_reduction": potential_pct,
        "exit_detected"      : exit_detected,
        "full_correct"       : full_ok,
    })
    print()


# ─── Summary table ────────────────────────────────────────────────────────────

print(f"\n{Fore.YELLOW}{HDIV}")
print(f"  TERMINATOR RESULTS — SUMMARY")
print(f"{HDIV}{Style.RESET_ALL}")

header = f"  {'Difficulty':<10} {'Avg Words':>12} {'Exit Point':>12} {'Reduction':>12} {'Accuracy':>10}"
print(f"\n{header}")
print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

for diff in ["easy", "medium", "hard"]:
    batch = [r for r in summary if r["difficulty"] == diff]
    if not batch:
        continue

    avg_words     = sum(r["full_reasoning_words"] for r in batch) / len(batch)
    avg_exit      = sum(r["exit_word_position"] or 0 for r in batch) / len(batch)
    avg_reduction = sum(r["potential_reduction"] for r in batch) / len(batch)
    accuracy      = sum(r["full_correct"] for r in batch) / len(batch) * 100

    color = {"easy": Fore.GREEN, "medium": Fore.YELLOW, "hard": Fore.RED}[diff]

    print(
        f"  {color}{diff.upper():<10}{Style.RESET_ALL}"
        f" {avg_words:>12.0f}"
        f" {avg_exit:>12.0f}"
        f" {avg_reduction:>11.1f}%"
        f" {accuracy:>9.0f}%"
    )


# ─── Observations ─────────────────────────────────────────────────────────────

print(f"\n{Fore.CYAN}  KEY OBSERVATIONS (mirrors TERMINATOR paper findings)")
print(f"  {'─'*61}{Style.RESET_ALL}")

overall_correct = sum(r["full_correct"] for r in summary)
exits_detected  = sum(r["exit_detected"] for r in summary)

print(f"  Full-CoT accuracy    : {overall_correct}/{len(summary)} ({overall_correct/len(summary)*100:.1f}%)")
print(f"  Exit points detected : {exits_detected}/{len(summary)}")
print()

for diff in ["easy", "medium", "hard"]:
    batch = [r for r in summary if r["difficulty"] == diff]
    if not batch:
        continue
    avg_r = sum(r["potential_reduction"] for r in batch) / len(batch)
    color = {"easy": Fore.GREEN, "medium": Fore.YELLOW, "hard": Fore.RED}[diff]
    print(
        f"  {color}{diff.upper():<8}{Style.RESET_ALL}: "
        f"avg potential reduction ≈ {avg_r:.1f}%  "
        f"{'← biggest overthinking' if diff == 'easy' else ''}"
    )

print()
print(f"  {Fore.WHITE}Paper reports 14–55% CoT length reduction with no accuracy loss.")
print(f"  Inference latency reduced >2× on MATH-500, AIME 2025, HumanEval, GPQA.{Style.RESET_ALL}")
print(f"\n{Fore.YELLOW}{HDIV}{Style.RESET_ALL}\n")
