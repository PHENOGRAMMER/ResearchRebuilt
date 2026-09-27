"""
main.py — Week 10 · LoRA: Low-Rank Adaptation
#ResearchRebuilt | Hu et al., ICLR 2022 | arxiv: 2106.09685

Entry point. Run the full paper recreation in sequence:
  Step 1: lora_math.py   — pure numpy, no API key needed
  Step 2: lora_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)

This file ties them together and prints the arc summary at the end.
"""

import subprocess, sys, os
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 62
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


banner("#ResearchRebuilt — Week 10 · LoRA (Hu et al., ICLR 2022)", Fore.CYAN)

print(f"""  Paper  : LoRA: Low-Rank Adaptation of Large Language Models
  Authors: Edward J. Hu, Yelong Shen, Phillip Wallis et al.  (Microsoft)
  Venue  : ICLR 2022  |  arxiv: 2106.09685
  Cited  : 11,000+ times — one of the most cited ML papers of the decade

  The idea in one sentence:
    Instead of updating all d×k weights in a transformer layer,
    freeze them and learn two small matrices B (d×r) and A (r×k)
    whose product BA is the update.  With r=8 on a 7B model:
    7,000,000,000 → 4,194,304 trainable parameters. Same accuracy.
""")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Running lora_math.py  (no API key needed)", Fore.YELLOW)
result = subprocess.run([sys.executable, "lora_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "lora_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Running lora_finetune.py  (Groq API)", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.")
    print("  Free key at https://console.groq.com\n")
else:
    subprocess.run([sys.executable, "lora_finetune.py"], capture_output=False)
