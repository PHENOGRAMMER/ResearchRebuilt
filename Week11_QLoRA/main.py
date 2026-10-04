"""
main.py — QLoRA: Efficient Finetuning of Quantized LLMs
#ResearchRebuilt | Dettmers et al., NeurIPS 2023 | arxiv: 2305.14314

Entry point. Run the paper recreation in sequence:
  Step 1: qlora_math.py    — pure numpy, no API key needed
  Step 2: qlora_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)
"""

import subprocess, sys, os
from colorama import Fore, Style, init
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
init(autoreset=True)


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 62
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


banner("QLoRA: Efficient Finetuning of Quantized LLMs", Fore.CYAN)

print(f"""  Paper       : QLoRA: Efficient Finetuning of Quantized LLMs
  Authors     : Tim Dettmers*, Artidoro Pagnoni*, Ari Holtzman, Luke Zettlemoyer
                (* equal contribution)
  Affiliation : University of Washington
  Venue       : NeurIPS 2023 — Oral presentation (top ~1% of submissions)
  arXiv       : https://arxiv.org/abs/2305.14314
  Impact      : Fine-tuned Guanaco 65B on a single 48 GB GPU in 24 hours;
                reached 99.3% of ChatGPT on the Vicuna benchmark.

  One-sentence summary:
    Quantise the frozen base model to 4-bit, keep the LoRA adapters in bf16,
    and backpropagate through the 4-bit weights into the adapters — so you
    get full fine-tuning accuracy at a fraction of the memory.

  Paper Brief:
    LoRA reduces trainable parameters dramatically, but the frozen base model
    still lives in full fp16 (~14 GB for 7B, ~130 GB for 65B). QLoRA quantises
    the base model to 4-bit while keeping adapters in full precision.

  The Three Innovations:
    1. NF4 (4-bit NormalFloat) — information-theoretically optimal for Gaussian weights
    2. Double Quantization     — quantise quantisation constants, saving ~0.37 bits/param
    3. Paged Optimizers        — page optimizer states to CPU to avoid long-sequence OOMs
""")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Running qlora_math.py  (no API key needed)", Fore.YELLOW)
result = subprocess.run([sys.executable, "qlora_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "qlora_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Running qlora_finetune.py  (Groq API)", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.")
    print("  Free key at https://console.groq.com\n")
else:
    subprocess.run([sys.executable, "qlora_finetune.py"], capture_output=False)

# ─────────────────────────────────────────────────────────────
# PAPER SUMMARY
# ─────────────────────────────────────────────────────────────
banner("QLORA SUMMARY & RECREATION COMPLETE", Fore.GREEN)

print(f"""  {Fore.GREEN}Key Takeaway:{Style.RESET_ALL}
    QLoRA enables high-quality fine-tuning of 65B parameter models on a single 48 GB GPU.
    By combining NF4 quantization, Double Quantization, and Paged Optimizers, QLoRA delivers
    99.3% of ChatGPT performance without losing model accuracy.

  {Fore.CYAN}Paper Link : https://arxiv.org/abs/2305.14314
  GitHub     : https://github.com/PHENOGRAMMER/ResearchRebuilt{Style.RESET_ALL}
""")
