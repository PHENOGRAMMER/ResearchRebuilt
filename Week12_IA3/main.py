"""
main.py — Week 12 · (IA)³: Few-Shot Parameter-Efficient Fine-Tuning
#ResearchRebuilt | Liu, Tam, Muqeeth et al., NeurIPS 2022 | arxiv: 2205.05638

Entry point. Run the full paper recreation in sequence:
  Step 1: ia3_math.py    — pure numpy, no API key needed
  Step 2: ia3_finetune.py — live Groq demo (needs GROQ_API_KEY in .env)

This file ties them together and prints the arc closure at the end.
"""

import subprocess, sys, os
from colorama import Fore, Style, init
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
init(autoreset=True)


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "=" * 62
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


banner("(IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations", Fore.CYAN)

print(f"""  Mechanism:
    Element-wise activation scaling via learned vectors l_k, l_v, and l_ff.
    Achieves parameter-efficient fine-tuning (~540K params for T0-3B) with zero
    inference latency overhead through direct weight re-scaling.

  Capabilities:
    • Ultra-low parameter footprint (~540K params)
    • Zero inference overhead via exact weight merging: W' = diag(l) @ W
    • Simultaneous mixed-task batching in a single forward pass
    • Outperforms full fine-tuning and massive prompt-based ICL with 1000×+ fewer FLOPs
""")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Math (always runs)
# ─────────────────────────────────────────────────────────────
banner("STEP 1 / 2 — Mathematical Validation & Mechanics", Fore.YELLOW)
result = subprocess.run([sys.executable, "ia3_math.py"], capture_output=False)
if result.returncode != 0:
    print(Fore.RED + "ia3_math.py failed — check the error above.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Live demo (gated on API key)
# ─────────────────────────────────────────────────────────────
banner("STEP 2 / 2 — Live Comparative Experiments", Fore.YELLOW)

if not os.getenv("GROQ_API_KEY"):
    print(Fore.YELLOW + "  GROQ_API_KEY not set → skipping live demo.")
    print("  Add GROQ_API_KEY=<key> to .env and re-run to see Experiments 1-3.")
else:
    subprocess.run([sys.executable, "ia3_finetune.py"], capture_output=False)

# ─────────────────────────────────────────────────────────────
# ARCHITECTURAL COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────
banner("EFFICIENCY COMPARISON — PEFT Paradigms", Fore.MAGENTA)

rows = [
    ("Paradigm", "Mechanism",          "Params",       "Base VRAM",    "Relative Perf"),
    ("─"*10,     "─"*20,               "─"*12,         "─"*13,         "─"*15),
    ("LoRA",     "Rank decomposition", "~4M  (r=8)",   "~14 GB fp16",  "Matches FT"),
    ("QLoRA",    "4-bit base + LoRA",  "~134M (r=64)", "~5 GB NF4",    "Matches FT"),
    ("(IA)³ ★",  "Scaling vectors",    "~540K",        "~6 GB (T0-3B)","Outperforms FT / ICL"),
]

col_w = [12, 22, 14, 15, 20]
for row in rows:
    line = "".join(str(c).ljust(col_w[i]) for i, c in enumerate(row))
    if row[0].startswith("─"):
        print(f"  {Fore.WHITE}{line}{Style.RESET_ALL}")
    elif row[0] == "Paradigm":
        print(f"  {Fore.CYAN}{Style.BRIGHT}{line}{Style.RESET_ALL}")
    elif "IA³" in row[0]:
        print(f"  {Fore.GREEN}{line}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}{line}{Style.RESET_ALL}")

print(f"""
  {Fore.GREEN}Conclusion: Element-wise scaling provides minimal trainable parameter count,
  enables mixed-task batching, and matches or exceeds full fine-tuning performance.{Style.RESET_ALL}
""")