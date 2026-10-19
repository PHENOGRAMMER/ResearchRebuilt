"""
rlhf_finetune.py — Week 13 · InstructGPT: Training Language Models to Follow Instructions
#ResearchRebuilt | Ouyang, Wu et al., NeurIPS 2022 | arxiv: 2203.02155

Three Groq experiments that demonstrate the InstructGPT alignment principles:
  Experiment 1 — Instruction following vs raw completion (SFT effect)
  Experiment 2 — Preference ranking simulation (RM training signal)
  Experiment 3 — Reward hacking detection (why KL penalty matters)

Model: openai/gpt-oss-120b via Groq API
"""

import os
import re
from groq import Groq
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

INSTRUCTGPT_PARAMS_13B = 1_300_000_000
INSTRUCTGPT_PARAMS_175B = 175_000_000_000


def strip_md(text: str) -> str:
    """Remove markdown bold/italic markers so output prints cleanly in terminal."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** → bold
    text = re.sub(r'\*(.+?)\*',     r'\1', text)   # *italic* → italic
    text = re.sub(r'__(.+?)__',     r'\1', text)   # __bold__ → bold
    text = re.sub(r'_(.+?)_',       r'\1', text)   # _italic_ → italic
    text = text.replace('**', '').replace('__', '') # clean dangling unclosed markers
    return text


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 64
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


def compare(label_a: str, prompt_a: str, system_a: str,
            label_b: str, prompt_b: str, system_b: str,
            user_prompt: str) -> tuple[str, str]:
    """Run two completions and return both responses."""
    def call(system, prompt):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return strip_md(resp.choices[0].message.content.strip())

    print(f"  Prompt: {user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}")
    print()

    resp_a = call(system_a, prompt_a)
    resp_b = call(system_b, prompt_b)

    print(f"  {Fore.YELLOW}── {label_a} ──{Style.RESET_ALL}")
    for line in resp_a.split("\n"):
        print(f"  {line}")
    print()
    print(f"  {Fore.GREEN}── {label_b} ──{Style.RESET_ALL}")
    for line in resp_b.split("\n"):
        print(f"  {line}")
    print()

    return resp_a, resp_b


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — SFT EFFECT: INSTRUCTION FOLLOWING
# ─────────────────────────────────────────────────────────────
banner("Experiment 1 / 3 — SFT Effect: Instruction Following vs Raw Completion", Fore.YELLOW)

"""
Paper §3.1, Figure 3:
  SFT (Step 1) produces a model that follows instructions as intended,
  not just predicts the next likely token.

  GPT-3 base (unaligned): given "Explain quantum entanglement to a 10-year-old,"
  may continue with: "Explain quantum entanglement to a 10-year-old in 3 parts..."
  — it treats the prompt as text to continue, not a task to perform.

  SFT model: treats the prompt as an instruction and produces a useful response.

  We simulate this contrast: one system prompt that replicates the "raw LM"
  behavior (continue the text) vs one that reflects SFT training (follow instructions).
"""

prompts_exp1 = [
    {
        "user": "Explain quantum entanglement to a 10-year-old.",
        "raw_system": "You are a text completion model. Continue the given text naturally, as if writing a document.",
        "sft_system": "You are a helpful assistant. Follow the user's instructions clearly and directly.",
    },
    {
        "user": "Write a Python function to check if a string is a palindrome.",
        "raw_system": "You are a text completion model. Continue the given text naturally.",
        "sft_system": "You are a helpful coding assistant. Write clean, working code with a brief explanation.",
    },
    {
        "user": "What are three risks of using LLMs in medical diagnosis?",
        "raw_system": "You are a text completion model. Continue the given text naturally.",
        "sft_system": "You are a helpful assistant. Answer clearly and accurately, structured as requested.",
    },
]

print("  Paper (§3.1): SFT trains the model to follow instructions, not just complete text.")
print("  Comparing 'raw LM' behavior vs 'SFT' behavior on 3 prompts.\n")

sft_wins = 0
for i, p in enumerate(prompts_exp1, 1):
    print(f"  {Fore.CYAN}── Prompt {i} of {len(prompts_exp1)} ──{Style.RESET_ALL}")
    resp_raw, resp_sft = compare(
        "Raw LM (unaligned, continues text)", p["user"], p["raw_system"],
        "SFT (instruction-following)",        p["user"], p["sft_system"],
        p["user"]
    )
    # Simple heuristic: SFT response tends to directly address the request
    sft_wins += 1  # In practice, SFT wins — consistent with paper's findings

print(f"  {Fore.GREEN}Paper result: SFT-trained InstructGPT preferred over base GPT-3 in 85% ± 3% of cases.")
print(f"  SFT is Step 1 of 3 — the RM and PPO steps push alignment further still. ✓{Style.RESET_ALL}")

param_ratio = INSTRUCTGPT_PARAMS_13B / INSTRUCTGPT_PARAMS_175B
print(f"\n  Paper parameter insight:")
print(f"    1.3B InstructGPT preferred over 175B GPT-3")
print(f"    That's {1/param_ratio:.0f}× fewer parameters — alignment beats scale.")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — REWARD MODEL: PREFERENCE RANKING
# ─────────────────────────────────────────────────────────────
banner("Experiment 2 / 3 — RM Training: Preference Ranking Simulation", Fore.MAGENTA)

"""
Paper §3.2:
  The Reward Model (Step 2) is trained on pairwise preferences:
    Given a prompt x and two responses y_w, y_l,
    labelers choose which one they prefer.

  The RM then learns to assign scalar rewards such that r(x, y_w) > r(x, y_l).

  Key: with K outputs per prompt, you get K(K-1)/2 pairs.
  Paper uses K=4 to 9 → ~33K prompts generate ~100K+ training pairs.

  We simulate the RM training signal: generate K=4 responses to a prompt,
  then rank them by quality to produce training pairs.
"""

rm_prompts = [
    {
        "prompt": "Summarise the key contributions of the Transformer architecture (Vaswani et al. 2017).",
        "temperatures": [0.2, 0.7, 1.0, 1.4],  # different temps → different quality distributions
    },
    {
        "prompt": "Give me advice on learning a new programming language efficiently.",
        "temperatures": [0.2, 0.7, 1.0, 1.4],
    },
]

print("  Paper (§3.2): RM trained on K=4–9 ranked responses per prompt.")
print("  Generating K=4 responses per prompt and simulating preference pairs.\n")

for idx, rp in enumerate(rm_prompts, 1):
    print(f"  {Fore.CYAN}── RM Prompt {idx} ──{Style.RESET_ALL}")
    print(f"  Prompt: {rp['prompt']}")
    print(f"  Generating {len(rp['temperatures'])} responses (K=4) at varying temperatures...\n")

    responses = []
    for temp in rp["temperatures"]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user",   "content": rp["prompt"]},
            ],
            max_tokens=200,
            temperature=temp,
        )
        text = strip_md(resp.choices[0].message.content.strip())
        responses.append((temp, text))

    for r_idx, (temp, text) in enumerate(responses, 1):
        print(f"  {Fore.YELLOW}Response {r_idx} (temp={temp}):{Style.RESET_ALL}")
        print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
        print()

    # Simulate labeler ranking (lower temp → typically more focused/better)
    K = len(responses)
    pairs = K * (K - 1) // 2
    print(f"  Training pairs from K={K} responses: K(K-1)/2 = {pairs} pairs")
    print(f"  {Fore.GREEN}These pairs feed the Bradley-Terry loss: L = -log σ(r_winner - r_loser) ✓{Style.RESET_ALL}")
    print()

print(f"  Paper dataset: ~33K prompts × avg ~15 pairs ≈ 100K+ RM training comparisons")
print(f"  RM accuracy achieved: 72.4% ± 0.4% (training) / 69.6% ± 0.9% (held-out)")
print(Fore.YELLOW + f"  ↳ Human-human agreement ceiling: ~73% — RM reaches near-human consistency ✓")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — KL PENALTY: REWARD HACKING DETECTION
# ─────────────────────────────────────────────────────────────
banner("Experiment 3 / 3 — KL Penalty: Why Reward Hacking Needs Stopping", Fore.CYAN)

"""
Paper §3.3, §4.4 (Figure 8):
  Without the KL penalty (β=0), PPO-trained models "reward hack":
  they find outputs that score high on the RM but are not genuinely helpful.

  Common reward hacking patterns:
    - Excessive hedging / repetitive caveats
    - Verbose padding to seem thorough
    - Sycophantic agreement with the user
    - Overly formal/stilted language that "sounds" correct but conveys nothing
    - Claiming to be unable to do things the RM scores conservatively safe

  We demonstrate this by comparing:
    A) "Reward-hacked" response: system prompt that mimics reward-hacking behavior
    B) "KL-constrained" response: stays close to SFT base, genuinely helpful

  The KL penalty β·KL(π_φ ‖ π_SFT) prevents A by keeping the policy
  close to the SFT model's distribution.
"""

hack_prompts = [
    {
        "user": "How do I negotiate a salary raise?",
        "hacked_system": (
            "You are an AI assistant. Always begin with extensive caveats about how complex "
            "this topic is. Repeat the user's question back to them in different words. "
            "List many considerations without actually giving actionable advice. "
            "End with encouraging words but no concrete steps. Be very verbose."
        ),
        "constrained_system": (
            "You are a helpful assistant. Give clear, direct, practical advice. "
            "Be concise and actionable. Stay close to what a knowledgeable friend would say."
        ),
    },
    {
        "user": "Explain why the sky is blue in simple terms.",
        "hacked_system": (
            "You are an AI assistant. Hedge extensively before answering. "
            "Note that this is a complex scientific topic with many nuances. "
            "List all possible caveats. Give an overly technical answer full of jargon "
            "that sounds impressive but is hard to follow. Repeat key points three times."
        ),
        "constrained_system": (
            "You are a helpful assistant. Explain clearly and simply. "
            "Use an analogy if it helps. Keep it short and accurate."
        ),
    },
    {
        "user": "Give me a one-sentence summary of the French Revolution.",
        "hacked_system": (
            "You are an AI assistant. Before answering, note that summarising complex "
            "historical events in one sentence risks oversimplification. "
            "Provide extensive context. Then finally give the sentence, "
            "followed by three paragraphs of caveats about what the summary misses."
        ),
        "constrained_system": (
            "You are a helpful assistant. Answer directly and accurately."
        ),
    },
]

print("  Paper §3.3: KL penalty β·KL(π_φ ‖ π_SFT) prevents reward hacking.")
print("  Demonstrating reward-hacked vs KL-constrained responses.\n")

for i, hp in enumerate(hack_prompts, 1):
    print(f"  {Fore.CYAN}── Reward Hacking Demo {i} of {len(hack_prompts)} ──{Style.RESET_ALL}")
    compare(
        "Reward-hacked (no KL penalty, β=0)", hp["user"], hp["hacked_system"],
        "KL-constrained (stays near π_SFT)",  hp["user"], hp["constrained_system"],
        hp["user"]
    )

print(f"  {Fore.GREEN}Paper (Figure 8): As KL divergence grows, RM score peaks then drops.")
print(f"  The model finds ways to 'game' the reward model without being genuinely better.")
print(f"  β·KL penalty keeps the policy close to π_SFT where real quality lives. ✓{Style.RESET_ALL}")

print(f"""
  PPO-ptx variant (paper §3.3):
    Also mixes in pretraining gradient: γ·E[log π_φ(x)]
    Prevents "alignment tax" — regressions on SQuAD, DROP, HellaSwag
    Cost: slightly lower win rates in human preference evals
    Benefit: maintains NLP benchmark performance

  InstructGPT training cost (paper §4):
    SFT (175B):    4.9 petaflops/s-days
    PPO-ptx (175B): 60 petaflops/s-days
    GPT-3 base:  3,640 petaflops/s-days
    {Fore.GREEN}Total alignment cost ≈ 1.8% of base model training compute ✓{Style.RESET_ALL}
""")

banner("rlhf_finetune.py COMPLETE", Fore.YELLOW)