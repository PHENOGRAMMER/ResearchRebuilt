"""
main.py — Week 16 · FlashAttention
#ResearchRebuilt | Dao, Fu, Ermon, Rudra, Ré · NeurIPS 2022 · arxiv: 2205.14135

Entry point. Runs flash_math.py (pure numpy, no API) then flash_finetune.py (Groq API).
"""

import subprocess
import sys
import os
def run(script: str) -> int:
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    return result.returncode


def main():
    scripts = [
        "flash_math.py",
        "flash_finetune.py",
    ]

    for script in scripts:
        path = os.path.join(os.path.dirname(__file__), script)
        if not os.path.exists(path):
            print(f"Missing: {script} — skipping.")
            continue
        rc = run(script)
        if rc != 0:
            print(f"{script} exited with code {rc}.")


if __name__ == "__main__":
    main()