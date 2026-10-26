"""
main.py — Week 14 · DPO: Direct Preference Optimization
#ResearchRebuilt | Rafailov, Sharma, Mitchell et al., NeurIPS 2023 | arxiv: 2305.18290

Entry point. Run the full paper recreation in sequence:
  Step 1: dpo_math.py    — pure numpy, no API key needed
  Step 2: dpo_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)

Alignment Arc: W13 InstructGPT → W14 DPO → W15 GRPO
"""

import subprocess, sys, os
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 64
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


banner("#ResearchRebuilt — Week 14 · DPO  (Rafailov et al., NeurIPS 2023)", Fore.CYAN)

print(f"""  Paper  : Direct Preference Optimization:
           Your Language Model is Secretly a Reward Model
  Authors: Rafael Rafailov*, Archit Sharma*, Eric Mitchell*,
           Stefano Ermon, Christopher D. Manning, Chelsea Finn
           — Stanford University
  Venue  : NeurIPS 2023
  Impact : Eliminates the reward model from RLHF.
           DPO becomes the dominant alignment recipe.

  The idea in one sentence:
    The optimal KL-constrained RL policy has a closed-form solution that
    expresses the reward function entirely in terms of the policy itself —
    so you can train directly on preferences, skipping the RM entirely.

  The three contributions:
    1. RLHF-to-DPO reduction  — closed-form derivation, Z(x) cancels
    2. DPO loss (Eq. 7)       — BT loss with implicit reward β·log(π/π_ref)
    3. Empirical results       — matches/exceeds PPO on sentiment, dialogue, summ.

""")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Running dpo_math.py  (no API key needed)", Fore.YELLOW)
result = subprocess.run([sys.executable, "dpo_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "dpo_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Running dpo_finetune.py  (Groq API)", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.\n")
else:
    subprocess.run([sys.executable, "dpo_finetune.py"], capture_output=False)

banner("Direct Preference Optimization (DPO) Pipeline Complete", Fore.GREEN)