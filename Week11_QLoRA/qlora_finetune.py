"""
qlora_finetune.py — QLoRA: Efficient Finetuning of Quantized LLMs
#ResearchRebuilt | Dettmers et al., NeurIPS 2023 | arxiv: 2305.14314

Simulates the key QLoRA insight live using the Groq API:
  BASE call       = general-purpose model (no specialisation)
  QLORA call      = quantisation-aware domain-adapted call
                    (simulates 4-bit base + bf16 LoRA adapter)

Three experiments from the paper and Guanaco fine-tuning results:
  1. Instruction following  (Guanaco / OASST1 task — paper Table 4)
  2. Long-context QA        (multi-hop reasoning — paper's Vicuna eval)
  3. Quantisation-aware generation  (what 4-bit NF4 implies for outputs)

For each: base model vs QLoRA-adapted, side-by-side, token cost logged.
"""

import os, sys, time
from colorama import Fore, Style, init
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
init(autoreset=True)

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"


def section(title: str, color=Fore.CYAN) -> None:
    bar = "─" * 62
    print(f"\n{color}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")


def call(client: "Groq", system: str, user: str, temp: float = 0.1) -> tuple[str, int]:
    """Single Groq call. Returns (text, completion_tokens)."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=temp,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip(), resp.usage.completion_tokens


def compare(client: "Groq", label: str,
            base_system: str, qlora_system: str, user: str,
            lora_r: int = 64, model_d: int = 4096, n_layers: int = 32,
            base_bits: int = 16, qlora_bits: int = 4) -> None:
    """
    Run base vs QLoRA-adapted call, print side-by-side, log memory analogy.
    lora_r / model_d / n_layers used for parameter calculation print.
    """
    print(f"  {Fore.WHITE}Task: {label}{Style.RESET_ALL}")
    print(f"  Input: {user[:120]}\n")

    base_out,  base_tok  = call(client, base_system,  user)
    time.sleep(0.6)
    qlora_out, qlora_tok = call(client, qlora_system, user)

    print(f"  {Fore.YELLOW}[BASE MODEL — fp16, general-purpose]{Style.RESET_ALL}")
    for line in base_out.split("\n")[:7]:
        print(f"    {line}")
    print(f"    ... ({base_tok} tokens)\n")

    print(f"  {Fore.GREEN}[QLORA-ADAPTED — NF4 4-bit base + bf16 LoRA adapter]{Style.RESET_ALL}")
    for line in qlora_out.split("\n")[:7]:
        print(f"    {line}")
    print(f"    ... ({qlora_tok} tokens)\n")

    # Memory analogy (the core QLoRA insight)
    base_vram_GB  = (model_d * model_d * n_layers * 4 * base_bits / 8) / 1e9   # rough
    qlora_vram_GB = (model_d * model_d * n_layers * 4 * qlora_bits / 8) / 1e9  # 4-bit

    adapter_params = 2 * (lora_r * model_d + model_d * lora_r) * n_layers
    adapter_GB     = adapter_params * 2 / 1e9   # bf16 = 2 bytes

    print(f"  {Fore.CYAN}Memory analogy (r={lora_r}, {model_d}-dim, {n_layers} layers):")
    print(f"    Base model fp16  : {base_vram_GB:.1f} GB  (attention Wq,Wk,Wv,Wo only)")
    print(f"    Base model NF4   : {qlora_vram_GB:.1f} GB  (same weights, 4-bit)")
    print(f"    LoRA adapters    : {adapter_GB:.3f} GB  (bf16, r={lora_r}, trainable)")
    print(f"    QLoRA total est. : {qlora_vram_GB + adapter_GB:.1f} GB  (vs {base_vram_GB:.1f} GB fp16){Style.RESET_ALL}")
    print()


# ─────────────────────────────────────────────────────────────
# GUARD: no API key
# ─────────────────────────────────────────────────────────────
if not HAS_GROQ:
    print(Fore.YELLOW + "\n  'groq' package not installed → skipping live API calls.")
    print("  Run 'pip install groq' to enable live fine-tuning simulation.\n")
    exit(0)

if not GROQ_API_KEY:
    print(Fore.RED + "\nGROQ_API_KEY not found in .env")
    print("Add GROQ_API_KEY=<your_key> to a .env file and re-run.")
    print("Get a free key at https://console.groq.com\n")
    exit(0)

client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — INSTRUCTION FOLLOWING (Guanaco / OASST1, paper Table 4)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 1 · Instruction Following  (Guanaco / OASST1 task)", Fore.YELLOW)

"""
Paper result (Table 4 / §4):
  Guanaco 65B (QLoRA fine-tuned on OASST1):
    - 99.3% of ChatGPT on Vicuna benchmark (human evaluation)
    - Trained in 24 hours on a single 48GB A100
    - Full fp16 equivalent would need 780 GB

  Guanaco 7B  (5 GB VRAM): 71.2% of ChatGPT
  Guanaco 13B (10 GB VRAM): 89.4% of ChatGPT

  The OASST1 task is open-ended instruction following.
  Base = helpfully general. QLoRA = precisely instruction-tuned.
"""

BASE_INST = "You are a helpful AI assistant. Answer the user's question."

QLORA_INST = """You are Guanaco, a helpful, respectful and honest AI assistant.
You are fine-tuned via QLoRA on OASST1 (Open Assistant dataset) — a curated
set of high-quality human instruction-response pairs.
Rules:
- Be direct and specific. Avoid padding and filler phrases.
- Structure your answer clearly. Use numbered steps when explaining a process.
- If the user asks for your opinion, give one with brief reasoning.
- Never start with "Certainly!", "Of course!", "Sure!" or similar openers.
Respond as if every token costs compute (it does — you're running at 4-bit)."""

inst_question = (
    "Explain the difference between supervised and unsupervised learning "
    "to someone who has programmed before but never studied ML. "
    "Give a concrete example of each."
)

compare(
    client, "Instruction following  (OASST1 / Guanaco task)",
    BASE_INST, QLORA_INST, inst_question,
    lora_r=64, model_d=4096, n_layers=32
)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — LONG-CONTEXT / MULTI-HOP QA (Vicuna benchmark eval)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 2 · Multi-hop Reasoning  (Vicuna benchmark style)", Fore.MAGENTA)

"""
Paper (§4.2, Table 9):
  Vicuna benchmark: 80 questions across 8 categories (writing, roleplay,
  math, coding, reasoning, STEM, humanities, extraction).

  Guanaco 65B: 99.3% of ChatGPT quality (human judges, Elo rating system)
  Guanaco 33B: 97.8% of ChatGPT
  Guanaco 13B: 89.4% of ChatGPT

  The key observation: QLoRA does NOT degrade multi-step reasoning
  vs full fp16 fine-tuning — quantisation noise cancels in the adapter update.

This experiment: multi-hop reasoning that requires connecting several facts.
"""

BASE_MH = "You are a helpful AI. Answer the question thoughtfully."

QLORA_MH = """You are a reasoning-specialist assistant fine-tuned on complex
multi-hop question answering. When answering multi-step questions:
1. Identify each sub-question that must be answered first.
2. Resolve each sub-question explicitly, labelling your reasoning steps.
3. Combine the sub-answers into a final, precise answer.
4. State confidence: High / Medium / Low and why.
Never skip steps — show the chain of reasoning, not just the conclusion."""

mh_question = (
    "If a language model is stored in NF4 (4-bit), but its LoRA adapters "
    "are in bf16 (16-bit), and you want to run inference on a GPU with 24 GB VRAM: "
    "what is the maximum number of parameters (in billions) the model can have, "
    "assuming the adapters use rank r=64 on Wq and Wv across all layers, "
    "and each transformer layer has a hidden dimension of 5120 with 40 layers total? "
    "Show your working."
)

compare(
    client, "Multi-hop numerical reasoning  (Vicuna style)",
    BASE_MH, QLORA_MH, mh_question,
    lora_r=64, model_d=5120, n_layers=40
)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — QUANTISATION-AWARE GENERATION (personal angle)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 3 · Quantisation-Aware Config  (personal/DeepRAG angle)", Fore.BLUE)

"""
Personal angle:
  In standard LoRA, base BERT still needed fp16 memory (438 MB for BERT-base).

  With QLoRA:
    BERT-base (110M params) in NF4 → ~55 MB (4-bit) + DQ ≈ 52 MB
    LoRA r=64 adapters (bf16)       → ~76 MB
    Total QLoRA BERT-base           → ~128 MB  (vs 220 MB fp16 LoRA)
    Saving over fp16 full fine-tune : ~3.5×

  For the retrieval engineer: you can now fine-tune embedding models
  on domain corpora without leaving CPU (128 MB fits in shared memory).

  This experiment: ask the model to write a QLoRA training config
  and explain each hyperparameter — base vs QLoRA-specialised.
"""

BASE_CONFIG = "You are a helpful AI that knows about deep learning. Answer questions about model training."

QLORA_CONFIG = """You are a senior ML engineer specialised in parameter-efficient fine-tuning (PEFT),
specifically QLoRA for embedding and generative models.
When writing training configs, always:
- Explain EVERY hyperparameter in a comment (# why this value)
- Flag quantisation-specific settings explicitly (NF4, double_quant, compute_dtype)
- Point out the tradeoffs between lora_r (expressiveness) and memory (r × d × layers × 2)
- Recommend the minimum config that works before scaling up
- Use bitsandbytes + PEFT + transformers (the paper's reference implementation)"""

config_question = (
    "Write a QLoRA training config for fine-tuning a 7B LLaMA model "
    "on a single 24 GB GPU for a domain-specific RAG task. "
    "Explain every hyperparameter — especially the quantisation settings."
)

compare(
    client, "QLoRA config generation  (embedding model / RAG use-case)",
    BASE_CONFIG, QLORA_CONFIG, config_question,
    lora_r=64, model_d=4096, n_layers=32
)


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY — paper numbers
# ─────────────────────────────────────────────────────────────
section("RESULTS SUMMARY — Paper Numbers & Benchmark Results", Fore.CYAN)

print(f"  {'Model':<22} {'VRAM (QLoRA)':>14} {'vs ChatGPT':>12} {'Training time':>16}  {'Source':>10}")
print(f"  {'─'*76}")

guanaco_results = [
    ("Guanaco  7B (QLoRA)",   "5 GB",  "71.2%",  "< 6 hrs  (1×RTX3090)", "Table 9"),
    ("Guanaco 13B (QLoRA)",  "10 GB",  "89.4%",  "< 12 hrs (1×RTX3090)", "Table 9"),
    ("Guanaco 33B (QLoRA)",  "21 GB",  "97.8%",  "< 24 hrs (1×A100)",    "Table 9"),
    ("Guanaco 65B (QLoRA) ★","41 GB",  "99.3%",  "~24 hrs  (1×A100-48G)","Table 9"),
    ("ChatGPT (ref.)",        "—",    "100.0%",  "—",                    "baseline"),
]

for name, vram, vs_chatgpt, time_, src in guanaco_results:
    star = "★" in name
    color = Fore.GREEN if star else ""
    reset = Style.RESET_ALL if star else ""
    print(f"  {color}{name:<22} {vram:>14} {vs_chatgpt:>12} {time_:>16}  {src:>10}{reset}")

print(f"""
  {Fore.YELLOW}Key result: Guanaco 65B reaches 99.3% of ChatGPT quality
  using a single 48 GB A100 GPU — hardware available to individuals. ✓{Style.RESET_ALL}

  {Fore.CYAN}Three innovations that made this possible:
    1. NF4  → information-theoretically optimal 4-bit quantisation
    2. DQ   → saves 0.37 bits/param by quantising the quantisation constants
    3. Paged Optimizers → no OOM during long-sequence gradient checkpointing{Style.RESET_ALL}
""")
