"""
main.py — Week 13 · InstructGPT: Training Language Models to Follow Instructions
#ResearchRebuilt | Ouyang, Wu et al., NeurIPS 2022 | arxiv: 2203.02155

Entry point. Run the full paper recreation in sequence:
  Step 1: rlhf_math.py    — pure numpy, no API key needed
  Step 2: rlhf_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)

Alignment Arc opener: W13 InstructGPT → W14 DPO → W15 GRPO
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


banner("#ResearchRebuilt — Week 13 · InstructGPT  (Ouyang et al., NeurIPS 2022)", Fore.CYAN)

print(f"""  Paper  : Training language models to follow instructions
           with human feedback
  Authors: Long Ouyang*, Jeff Wu*, Xu Jiang, Diogo Almeida,
           Carroll Wainwright, Pamela Mishkin, Chong Zhang,
           Sandhini Agarwal, Katarina Slama, Alex Ray,
           John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
           Maddie Simens, Amanda Askell, Peter Welinder,
           Paul Christiano, Jan Leike, Ryan Lowe — OpenAI
  Venue  : NeurIPS 2022
  Impact : 1.3B InstructGPT preferred over 175B GPT-3.
           Alignment beats scale. RLHF becomes the standard.

  The idea in one sentence:
    Fine-tune GPT-3 in three steps — supervised on demonstrations (SFT),
    then train a reward model on human preference rankings (RM),
    then PPO-optimize against it with a KL penalty to prevent reward hacking.
    The result follows user intent, not next-token probability.

  The three contributions:
    1. SFT model          — demonstration-trained, instruction-following base
    2. Reward model (RM)  — 6B scalar scorer trained on 33K ranked pairs
    3. PPO-ptx            — RL + KL penalty + pretraining mix = InstructGPT
""")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Running rlhf_math.py  (no API key needed)", Fore.YELLOW)
result = subprocess.run([sys.executable, "rlhf_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "rlhf_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Running rlhf_finetune.py  (Groq API)", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.\n")
else:
    subprocess.run([sys.executable, "rlhf_finetune.py"], capture_output=False)

banner("InstructGPT (RLHF) Pipeline Complete", Fore.GREEN)