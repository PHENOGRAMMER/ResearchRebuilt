"""
main.py — Week 15 · GRPO: Group Relative Policy Optimization
#ResearchRebuilt | Shao, Wang, Zhu et al., 2024 | arxiv: 2402.03300
DeepSeek-Math: Pushing the Limits of Mathematical Reasoning in LLMs

Entry point. Run the full paper recreation in sequence:
  Step 1: grpo_math.py    — pure numpy, no API key needed
  Step 2: grpo_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)
"""

import subprocess, sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 64
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Running grpo_math.py  (no API key needed)", Fore.YELLOW)
result = subprocess.run([sys.executable, "grpo_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "grpo_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Running grpo_finetune.py  (Groq API)", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.")
    print("  Free key at https://console.groq.com\n")
else:
    subprocess.run([sys.executable, "grpo_finetune.py"], capture_output=False)