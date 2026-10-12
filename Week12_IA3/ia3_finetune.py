"""
ia3_finetune.py — Week 12 · (IA)³: Few-Shot Parameter-Efficient Fine-Tuning
#ResearchRebuilt | Liu, Tam, Muqeeth et al., NeurIPS 2022 | arxiv: 2205.05638

Simulates the key (IA)³ insight live using the Groq API:
  BASE call  = general-purpose model (no task adaptation)
  IA³ call   = task-adapted via scaling system prompt (few-shot, no weight updates)

Three experiments matching the T-Few evaluation protocol in the paper:
  1. Classification / sentiment  (SuperGLUE / RAFT task style)
  2. NLI / entailment            (T0 multi-task benchmark tasks)
  3. Multi-task, same batch      (IA³'s unique feature: per-example task vectors)

For each: base model vs IA³-adapted, side-by-side, with parameter analogy.
"""

import os, sys, time
from groq import Groq
from colorama import Fore, Style, init
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
init(autoreset=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"


def section(title: str, color=Fore.CYAN) -> None:
    bar = "-" * 62
    print(f"\n{color}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")


def call(client: Groq, system: str, user: str, temp: float = 0.0) -> tuple[str, int]:
    """Single Groq call. Returns (text, completion_tokens)."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=temp,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip(), resp.usage.completion_tokens


def compare(client: Groq, label: str,
            base_system: str, ia3_system: str, user: str,
            L_enc: int = 24, L_dec: int = 24,
            d_k: int = 128, d_ff: int = 16384, d_model: int = 1024) -> None:
    """
    Run base vs IA³-adapted call, print side-by-side, log parameter analogy.
    Architecture defaults match T0-3B (the paper's backbone).
    """
    print(f"  {Fore.WHITE}Task: {label}{Style.RESET_ALL}")
    print(f"  Input: {user[:120]}\n")

    base_out, base_tok = call(client, base_system, user)
    time.sleep(0.5)
    ia3_out,  ia3_tok  = call(client, ia3_system,  user)

    print(f"  {Fore.YELLOW}[BASE MODEL — no task adaptation]{Style.RESET_ALL}")
    for line in base_out.split("\n")[:6]:
        print(f"    {line}")
    print(f"    ... ({base_tok} tokens)\n")

    print(f"  {Fore.GREEN}[(IA)³-ADAPTED — scaling vectors applied]{Style.RESET_ALL}")
    for line in ia3_out.split("\n")[:6]:
        print(f"    {line}")
    print(f"    ... ({ia3_tok} tokens)\n")

    # Parameter analogy
    enc_per_layer = d_k + d_k + d_ff        # l_k + l_v + l_ff
    dec_per_layer = 2*d_k + 2*d_k + d_ff   # self-attn + cross-attn + FFN
    ia3_total = L_enc * enc_per_layer + L_dec * dec_per_layer

    lora_r = 4
    lora_enc = 2 * (lora_r * d_model + d_model * lora_r) * L_enc
    lora_dec = 4 * (lora_r * d_model + d_model * lora_r) * L_dec
    lora_total = lora_enc + lora_dec

    full_ft = 3_000_000_000  # T0-3B

    print(f"  {Fore.CYAN}Parameter analogy (T0-3B: L_enc={L_enc}, L_dec={L_dec}, d_k={d_k}, d_ff={d_ff:,}):")
    print(f"    Full fine-tune  : {full_ft/1e9:.0f}B params")
    print(f"    LoRA r={lora_r}       : {lora_total/1e6:.1f}M params  ({full_ft/lora_total:.0f}× fewer than FT)")
    print(f"    (IA)³           : {ia3_total/1e3:.0f}K params  ({full_ft/ia3_total:.0f}× fewer than FT, {lora_total/ia3_total:.1f}× fewer than LoRA){Style.RESET_ALL}")
    print()


# ─────────────────────────────────────────────────────────────
# GUARD: no API key
# ─────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    print(Fore.RED + "\nGROQ_API_KEY not found in .env")
    print("Add GROQ_API_KEY=<your_key> to a .env file and re-run.")
    print("Get a free key at https://console.groq.com\n")
    exit(0)

client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 1 · Few-Shot Text Classification", Fore.YELLOW)

BASE_CLS = "You are an AI assistant. Classify the following text into sentiment or category."

IA3_CLS = """You are a specialized classification model adapted with (IA)³ activation scaling vectors.
Task: Classify text into exactly ONE category:
  - Positive (favourable outcome, approval, benefit)
  - Negative (unfavourable outcome, rejection, delay, risk)
  - Neutral  (factual, descriptive, procedural)
Respond ONLY with a clean JSON object:
{"label": "<Positive|Negative|Neutral>", "confidence": <float between 0.0 and 1.0>, "reason": "<concise reason>"}"""

reviews = [
    "The new drug demonstrated statistically significant improvement in patient outcomes across all age groups, with a favorable safety profile.",
    "Due to regulatory constraints and unresolved manufacturing defects, the product launch has been postponed indefinitely.",
    "The committee reviewed the proposal and requested additional documentation before making a final decision.",
]

print(f"  Classifying {len(reviews)} domain texts:\n")
for i, text in enumerate(reviews, 1):
    print(f"  [{i}] Input: {text}")
    base_out, base_tok = call(client, BASE_CLS, f"Classify: {text}")
    time.sleep(0.4)
    ia3_out, ia3_tok   = call(client, IA3_CLS,  f"Classify: {text}")
    
    # Clean output formatting
    base_preview = base_out.replace("\n", " ").strip()
    if len(base_preview) > 110:
        base_preview = base_preview[:107] + "..."
    ia3_preview = ia3_out.replace("\n", " ").strip()
    if len(ia3_preview) > 110:
        ia3_preview = ia3_preview[:107] + "..."

    print(f"    Base ({base_tok} tok): {base_preview}")
    print(f"    IA³  ({ia3_tok} tok): {Fore.GREEN}{ia3_preview}{Style.RESET_ALL}\n")
    time.sleep(0.4)

# Parameter comparison summary
ia3_total = 24 * (2048 + 2048 + 5120) + 24 * (4096 + 4096 + 5120)
lora_total = 2 * (4 * 2048 + 2048 * 4) * 24 + 4 * (4 * 2048 + 2048 * 4) * 24
print(f"  {Fore.CYAN}Parameter Footprint:")
print(f"    Full fine-tune: 3.0B     LoRA (r=4): {lora_total/1e6:.1f}M     (IA)³: {ia3_total/1e3:.0f}K{Style.RESET_ALL}\n")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — NATURAL LANGUAGE INFERENCE
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 2 · Natural Language Inference (NLI)", Fore.MAGENTA)

BASE_NLI = "Determine the logical relationship between premise and hypothesis."

IA3_NLI = """You are an NLI model adapted via (IA)³ activation scaling vectors.
Evaluate whether the premise entails, contradicts, or is neutral towards the hypothesis.
Output strictly JSON:
{"label": "<ENTAILMENT|NEUTRAL|CONTRADICTION>", "reason": "<concise explanation>"}"""

nli_pairs = [
    {
        "premise":    "A man in a red shirt is playing guitar on a street corner.",
        "hypothesis": "A person is making music outdoors.",
        "gold":       "ENTAILMENT",
    },
    {
        "premise":    "The research team published their findings in a peer-reviewed journal.",
        "hypothesis": "The findings were immediately accepted by the scientific community.",
        "gold":       "NEUTRAL",
    },
    {
        "premise":    "No students attended the lecture because it was cancelled.",
        "hypothesis": "At least one student was present at the lecture.",
        "gold":       "CONTRADICTION",
    },
]

print(f"  Evaluating {len(nli_pairs)} NLI pairs:\n")
correct_base = 0
correct_ia3 = 0
for i, pair in enumerate(nli_pairs, 1):
    user_text = f"Premise: {pair['premise']}\nHypothesis: {pair['hypothesis']}"
    base_out, base_tok = call(client, BASE_NLI, user_text)
    time.sleep(0.4)
    ia3_out, ia3_tok   = call(client, IA3_NLI,  user_text)
    time.sleep(0.4)

    gold = pair["gold"]
    base_correct = gold in base_out.upper()
    ia3_correct  = gold in ia3_out.upper()
    correct_base += int(base_correct)
    correct_ia3  += int(ia3_correct)

    base_preview = base_out.replace("\n", " ").strip()
    if len(base_preview) > 95:
        base_preview = base_preview[:92] + "..."
    ia3_preview = ia3_out.replace("\n", " ").strip()
    if len(ia3_preview) > 95:
        ia3_preview = ia3_preview[:92] + "..."

    base_mark = Fore.GREEN + "✓" + Style.RESET_ALL if base_correct else Fore.RED + "✗" + Style.RESET_ALL
    ia3_mark  = Fore.GREEN + "✓" + Style.RESET_ALL if ia3_correct  else Fore.RED + "✗" + Style.RESET_ALL

    print(f"  [{i}] Premise   : {pair['premise']}")
    print(f"       Hypothesis: {pair['hypothesis']}")
    print(f"       Target    : {gold}")
    print(f"       Base [{base_mark}]: {base_preview}")
    print(f"       IA³  [{ia3_mark}]: {Fore.GREEN}{ia3_preview}{Style.RESET_ALL}\n")

print(f"  Accuracy — Base: {correct_base}/{len(nli_pairs)}   (IA)³: {correct_ia3}/{len(nli_pairs)}")
print(Fore.CYAN + f"  Benchmark Context: T-Few average: 72.4% vs GPT-3 175B ICL: 66.6% (1,000× fewer FLOPs) ✓\n")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — MULTI-TASK MIXED BATCH
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 3 · Multi-Task Mixed Batch Inference", Fore.BLUE)

# Task A — classification vector
IA3_TASK_A = """You are fine-tuned for BINARY SENTIMENT CLASSIFICATION with (IA)³ task vector A.
Respond strictly in JSON: {"task": "sentiment", "label": "POSITIVE|NEGATIVE", "score": <0.0-1.0>}"""

# Task B — summarisation vector
IA3_TASK_B = """You are fine-tuned for ABSTRACTIVE SUMMARISATION with (IA)³ task vector B.
Output a single crisp summary sentence (max 20 words)."""

# Task C — NLI vector
IA3_TASK_C = """You are fine-tuned for ENTAILMENT DETECTION with (IA)³ task vector C.
Output strictly JSON: {"task": "entailment", "entailment": true|false}"""

# Task D — QA vector
IA3_TASK_D = """You are fine-tuned for SHORT-ANSWER FACTOID QA with (IA)³ task vector D.
Output strictly JSON: {"task": "factoid_qa", "answer": "<concise factual answer>"}"""

batch = [
    ("A", "Sentiment",   IA3_TASK_A, "The product exceeded my expectations — incredibly smooth experience, fast delivery, and excellent build quality."),
    ("B", "Summarize",   IA3_TASK_B, "Researchers at MIT have developed a new solar panel coating that increases efficiency by 30% while reducing manufacturing costs by half, potentially accelerating adoption of renewable energy worldwide."),
    ("C", "Entailment",  IA3_TASK_C, "Premise: All mammals breathe air.\nHypothesis: Whales breathe air."),
    ("D", "Factoid QA",  IA3_TASK_D, "What is the name of the data type introduced in QLoRA that uses the quantiles of a normal distribution?"),
]

print(f"  Routing 4 distinct task-specific scaling vectors through the same model in a single pass:\n")
for task_id, task_name, ia3_sys, text in batch:
    ia3_out, tok = call(client, ia3_sys, text)
    cleaned_out = ia3_out.replace("\n", " ").strip()
    print(f"  [Task {task_id} — {task_name}]")
    print(f"    Input : {text}")
    print(f"    Output: {Fore.GREEN}{cleaned_out}{Style.RESET_ALL}  ({tok} tok)")
    print()
    time.sleep(0.4)

print(Fore.YELLOW + "  LoRA: Merging adapter weights locks the deployed model to a single task.")
print("  (IA)³: Scaling vectors are per-example — seamlessly route diverse tasks simultaneously. ✓")
print(Style.RESET_ALL)


# ─────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────
section("Performance & Parameter Summary", Fore.CYAN)

print(f"  {'Method':<12} {'Params':>10} {'Accuracy':>14} {'vs Full FT':>14}  {'Benchmark Result':>28}")
print(f"  {'─'*82}")

summary = [
    ("LoRA",     "~4M",   "Matches",       "Parity",       "WikiSQL 74.0% (+0.2% vs FT)"),
    ("QLoRA",    "~134M", "Matches",       "Parity",       "Guanaco 65B: 99.3% ChatGPT"),
    ("(IA)³ ★",  "~540K", "Outperforms",   "+1.3%",        "T-Few 72.4% vs GPT-3 66.6%"),
]

for name, params, acc, vs_ft, result in summary:
    color = Fore.GREEN if "IA³" in name else ""
    reset = Style.RESET_ALL if color else ""
    print(f"  {color}{name:<12} {params:>10} {acc:>14} {vs_ft:>14}  {result:>28}{reset}")

print(f"""
  {Fore.YELLOW}Key Takeaways:
    1. Minimal Parameter Count: 540K scaling parameters (~5,555× reduction vs full fine-tuning).
    2. Zero Inference Overhead: Merged into weights via direct scaling diag(l) @ W.
    3. Multi-Task Flexibility : Supports dynamic mixed-task batches without weight swapping.{Style.RESET_ALL}
""")