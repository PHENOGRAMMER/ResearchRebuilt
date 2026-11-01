"""
grpo_finetune.py — Week 15 · GRPO: Group Relative Policy Optimization
#ResearchRebuilt | Shao, Wang, Zhu et al., 2024 | arxiv: 2402.03300

Three Groq experiments that demonstrate GRPO's core principles:
  Experiment 1 — Group sampling: G responses per prompt, group advantage computation
  Experiment 2 — Outcome reward signal: binary and scalar rewards, group normalisation
  Experiment 3 — Value model vs group baseline: what each method sees as "good"

Model: openai/gpt-oss-120b via Groq API
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import re
import numpy as np
from groq import Groq
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "openai/gpt-oss-120b"

G    = 4      # group size — paper uses 8; we use 4 to limit API calls
BETA = 0.04   # KL penalty — paper default for math tasks
EPS  = 0.2    # PPO clip ratio


def strip_md(text: str) -> str:
    """Remove markdown bold/italic markers, carriage returns, and LaTeX delimiters."""
    text = text.replace("\r", "")
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'__(.+?)__',     r'\1', text)
    text = re.sub(r'_(.+?)_',       r'\1', text)
    text = text.replace(r"\$", "$")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    return text.strip()


def banner(text: str, color=Fore.CYAN) -> None:
    bar = "═" * 64
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{Style.RESET_ALL}\n")


def call(system: str, user: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
    """Single Groq call, markdown and LaTeX stripped."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        return strip_md(content)
    except Exception as e:
        return f"[API Error: {e}]"


def group_advantage(rewards: np.ndarray) -> np.ndarray:
    """
    GRPO normalised advantage (paper Eq. 11):
      Â_i = (r_i - mean(r)) / std(r)
    This is the group baseline. No value model.
    """
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards) + 1e-8
    return (rewards - mean_r) / std_r


def score_math(response: str, expected: str) -> float:
    """
    Binary outcome reward for math:
      1.0 if response contains the correct answer
      0.0 otherwise
    In real GRPO training this is a verified math checker.
    """
    resp_clean = response.lower().replace(",", "").replace(" ", "").replace("$", "")
    exp_clean  = str(expected).lower().replace(",", "").replace(" ", "").replace("$", "")
    return 1.0 if exp_clean in resp_clean else 0.0


def score_quality(response: str, prompt: str) -> float:
    """
    Scalar quality score proxy (0–1):
    Scores responses based on clarity, structure, conciseness, and absence of filler/noise.
    In real GRPO training, this score comes from a learned reward model or verifier.
    """
    lower = response.lower()
    words = response.split()
    n_words = len(words)

    # 1. Base score from length / substance
    if n_words < 25:
        base = 0.25
    elif n_words <= 160:
        base = 0.85
    elif n_words <= 220:
        base = 0.70
    else:
        base = 0.40

    # 2. Filler / noise penalty (characteristic of low-quality generation)
    fillers = [
        "you know", "kind of", "sort of", "uh", "um", "like, you",
        "well, okay", "basically", "wander", "thingy", "take my time",
        "looping", "circling back", "right?", "what you asked for"
    ]
    filler_count = sum(lower.count(f) for f in fillers)
    filler_penalty = min(filler_count * 0.15, 0.60)

    # 3. Structure bonus (numbered lists, clear formatting)
    structure_bonus = 0.0
    if bool(re.search(r'\d+[\.\)]\s', response)) or "1." in response or "##" in response:
        structure_bonus += 0.12
    if response.count("\n") >= 2:
        structure_bonus += 0.05

    # 4. Technical content bonus (uses core GRPO concepts)
    keywords = ["baseline", "group", "mean", "critic", "value model", "reward", "token", "trajectory"]
    keyword_matches = sum(1 for kw in keywords if kw in lower)
    content_bonus = min(keyword_matches * 0.03, 0.12)

    # Combine with small noise for slight variation
    noise = np.random.uniform(-0.02, 0.02)
    score = base - filler_penalty + structure_bonus + content_bonus + noise
    return float(np.clip(score, 0.10, 1.00))


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — GROUP SAMPLING + ADVANTAGE COMPUTATION
# ─────────────────────────────────────────────────────────────
banner("Experiment 1 / 3 — Group Sampling: G Responses, Group Advantage", Fore.YELLOW)

print(f"""  Paper §3: GRPO samples G responses per prompt from the current policy.
  The group mean reward becomes the baseline — no value model.
  Advantage: Â_i = (r_i - mean(r)) / std(r)

  Sampling G={G} scripted rollouts per prompt — correct reasoning paths
  and common error types — to simulate on-policy rollout diversity.
""")

# Pre-defined rollouts: fixed texts that show realistic on-policy diversity.
# In real GRPO training these come from the policy at different training steps.
# Using fixed rollouts here ensures deterministic, readable advantage tables.
math_prompts = [
    {
        "user": (
            "Solve step by step: A merchant buys goods for $480 and sells them "
            "at a profit such that the selling price is 5/4 of the cost. "
            "He then gives a 10% discount on the selling price. "
            "What is the final selling price?"
        ),
        "answer": "540",
        "reward_type": "binary (math correct/wrong)",
        "rollouts": [
            # y_1: correct — algebraic chain
            (
                "correct chain",
                "Selling price = 5/4 * 480 = 600.\n"
                "10% discount on 600 = 60.\n"
                "Final selling price = 600 - 60 = 540.\n"
                "Final answer: 540"
            ),
            # y_2: correct — decimal shortcut
            (
                "correct shortcut",
                "SP before discount = 1.25 * 480 = 600.\n"
                "After 10% discount: 600 * 0.90 = 540.\n"
                "Final answer: 540"
            ),
            # y_3: wrong — subtracts 10% of cost, not SP
            (
                "error: discount on cost",
                "Selling price = 5/4 * 480 = 600.\n"
                "10% of cost = 0.10 * 480 = 48.\n"
                "Final price = 600 - 48 = 552.\n"
                "Final answer: 552"
            ),
            # y_4: wrong — forgets to apply discount
            (
                "error: discount omitted",
                "The selling price is 5/4 of the cost.\n"
                "SP = 5/4 * 480 = 600.\n"
                "Final answer: 600"
            ),
        ],
    },
    {
        "user": (
            "Solve step by step: A tank is 3/5 full. "
            "After removing 24 litres it is 2/5 full. "
            "What is the total capacity of the tank in litres?"
        ),
        "answer": "120",
        "reward_type": "binary (math correct/wrong)",
        "rollouts": [
            # y_1: correct
            (
                "correct — fraction difference",
                "3/5 - 2/5 = 1/5 of the tank was removed.\n"
                "1/5 * capacity = 24 litres.\n"
                "Capacity = 24 * 5 = 120.\n"
                "Final answer: 120"
            ),
            # y_2: correct alternate
            (
                "correct — algebra",
                "Let capacity = C.\n"
                "3C/5 - 24 = 2C/5.\n"
                "C/5 = 24, so C = 120.\n"
                "Final answer: 120"
            ),
            # y_3: wrong — uses 1/3 instead of 1/5
            (
                "error: wrong fraction (1/3)",
                "The tank went from 3/5 to 2/5.\n"
                "I'll treat the removed amount as 1/3 of the tank.\n"
                "Capacity = 24 / (1/3) = 72.\n"
                "Final answer: 72"
            ),
            # y_4: wrong — adds fractions instead of subtracting
            (
                "error: adds fractions",
                "3/5 + 2/5 = 5/5 = full tank = 24 litres.\n"
                "Capacity = 24.\n"
                "Final answer: 24"
            ),
        ],
    },
]

for pidx, p in enumerate(math_prompts, 1):
    print(f"  {Fore.CYAN}── Math Prompt {pidx} ──{Style.RESET_ALL}")
    print(f"  Prompt: {p['user']}")
    print(f"  Correct answer: {p['answer']}")
    print(f"  Reward type: {p['reward_type']}\n")
    print(f"  G={G} rollouts — 2 correct reasoning paths, 2 common errors:\n")

    responses = []
    rewards   = []

    for i, (label, resp_text) in enumerate(p["rollouts"]):
        reward = score_math(resp_text, p["answer"])
        responses.append((label, resp_text))
        rewards.append(reward)
        print(f"  {Fore.YELLOW}y_{i+1} [{label}]:{Style.RESET_ALL}")
        for line in resp_text.strip().split("\n"):
            print(f"    {line}")
        print(f"  Outcome reward: {Fore.GREEN if reward == 1.0 else Fore.RED}{reward:.1f}{Style.RESET_ALL} "
              f"({'correct ✓' if reward == 1.0 else 'wrong ✗'})\n")

    rewards_arr = np.array(rewards)
    advantages  = group_advantage(rewards_arr)

    print(f"  {Fore.GREEN}GRPO group advantage computation:{Style.RESET_ALL}")
    print(f"  Group mean reward: {np.mean(rewards_arr):.3f}  |  Group std: {np.std(rewards_arr):.3f}")
    print()
    print(f"  {'Response':<12} {'Reward':>8} {'Â_i':>10}  {'Policy update direction'}")
    print(f"  {'─'*55}")
    for i, (r, a) in enumerate(zip(rewards_arr, advantages)):
        direction = "increase prob ↑" if a > 0 else ("decrease prob ↓" if a < 0 else "no change →")
        color = Fore.GREEN if a > 0 else (Fore.RED if a < 0 else Fore.WHITE)
        print(f"  {color}y_{i+1:<9} {r:>8.1f} {a:>10.4f}  {direction}{Style.RESET_ALL}")
    print(f"\n  ↳ No value model used. Group mean = baseline. ✓")
    print(f"  ↳ Wrong answers get negative advantage → probability decreases ✓\n")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — OUTCOME REWARD SIGNAL
# ─────────────────────────────────────────────────────────────
banner("Experiment 2 / 3 — Outcome Reward Signal: Binary vs Scalar Rewards", Fore.MAGENTA)

print(f"""  Paper §3.2: GRPO is designed for outcome-based rewards.
  Two reward types:
    Binary (math/code):  r ∈ {{0, 1}} — correct or wrong
    Scalar (quality RM): r ∈ [0, 1]  — scored by a reward model

  Both use the same GRPO group advantage formula.
  The key: reward is assigned to the FULL RESPONSE, not per token.
  This is what makes the value model unnecessary.
""")

quality_prompts = [
    {
        "user": "Explain the intuition behind why GRPO doesn't need a value model.",
        "reward_type": "scalar quality score",
        "systems": [
            "You are a precise ML explainer. In 3 clear sentences, explain why GRPO eliminates the value model by using the group mean reward as the baseline. Be direct and accurate.",
            "You are extremely wordy and unfocused. Use lots of filler words like 'you know', 'kind of', 'sort of', 'uh', 'um', 'well okay'. Repeat yourself constantly, wander off-topic, and be as vague as possible.",
            "You are a helpful ML tutor. Give a clear explanation in 2 numbered steps: 1) Group baseline replaces the critic, 2) Outcome rewards eliminate per-token TD errors. Add a one-line example.",
            "Answer in only one short sentence of at most 8 words.",
        ],
        "labels": ["precise", "verbose+noisy", "structured+example", "too brief"],
    },
]

np.random.seed(7)

for p in quality_prompts:
    print(f"  {Fore.CYAN}── Quality Scoring Demo ──{Style.RESET_ALL}")
    print(f"  Prompt: {p['user']}")
    print(f"  Reward type: {p['reward_type']}\n")

    responses = []
    rewards   = []

    for sys_prompt, label in zip(p["systems"], p["labels"]):
        resp   = call(sys_prompt, p["user"], temperature=0.7)
        reward = score_quality(resp, p["user"])
        responses.append((label, resp))
        rewards.append(reward)

    rewards_arr = np.array(rewards)
    advantages  = group_advantage(rewards_arr)

    # Print responses
    for i, ((label, resp), r) in enumerate(zip(responses, rewards_arr)):
        print(f"  {Fore.YELLOW}y_{i+1} [{label}] — reward: {r:.3f}{Style.RESET_ALL}")
        lines = [l.strip() for l in resp.replace("\r", "").split("\n") if l.strip()]
        for line in lines[:4]:
            print(f"    {line}")
        if len(lines) > 4:
            print(f"    ...")
        print()

    print(f"  {Fore.GREEN}Group advantage table:{Style.RESET_ALL}")
    print(f"  Group mean: {np.mean(rewards_arr):.3f}  |  Group std: {np.std(rewards_arr):.3f}")
    print()
    print(f"  {'Label':<22} {'Reward':>8} {'Â_i':>10}  Update direction")
    print(f"  {'─'*60}")
    for (label, _), r, a in zip(responses, rewards_arr, advantages):
        direction = "↑ increase" if a > 0.2 else ("↓ decrease" if a < -0.2 else "→ neutral")
        color = Fore.GREEN if a > 0.2 else (Fore.RED if a < -0.2 else Fore.WHITE)
        print(f"  {color}{label:<22} {r:>8.3f} {a:>10.4f}  {direction}{Style.RESET_ALL}")

    print(f"""
  Paper §3: GRPO with outcome-based rewards achieves stable training
  because reward variance is controlled by group normalisation, not by
  a critic that must approximate future returns token-by-token.

  Binary vs scalar: both work because Â_i = (r - μ) / σ normalises
  any reward scale into a zero-mean unit-variance signal. ✓
""")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — VALUE MODEL VS GROUP BASELINE
# ─────────────────────────────────────────────────────────────
banner("Experiment 3 / 3 — Value Model vs Group Baseline: What Each Method Sees", Fore.CYAN)

print(f"""  Paper §3 (contrast with PPO):
  PPO's value model estimates: "How good is it to be at token t?"
    V(s_t) ≈ E[ Σ_{{t'≥t}} r_{{t'}} | s_t ]   ← a number for every token

  GRPO's group baseline estimates: "How good is this response vs the group?"
    baseline(x) = mean({{r_1, ..., r_G}})       ← one number per prompt

  The question: does per-token credit assignment (PPO) actually help
  for math reasoning tasks where the reward is at the END?

  Paper finding: for outcome-based tasks, GRPO matches PPO on MATH/GSM8K
  while using 0 extra parameters for the critic.

  We demonstrate the difference by showing what each method would do
  with three qualitatively different responses to the same prompt.
""")

# Fixed rollouts for Exp 3 — same rationale as Exp 1.
# Three qualitatively different responses to the same prompt.
reasoning_prompts = [
    {
        "user": "A store sells apples for $0.50 each and oranges for $0.75 each. If Maya buys 6 apples and 4 oranges, how much does she spend in total?",
        "answer": "6.00",
        "rollouts": [
            # y_1: correct, well-structured
            (
                "clean reasoning",
                "Step 1: Cost of apples = 6 * $0.50 = $3.00\n"
                "Step 2: Cost of oranges = 4 * $0.75 = $3.00\n"
                "Step 3: Total = $3.00 + $3.00 = $6.00\n"
                "Maya spends $6.00 in total."
            ),
            # y_2: correct, terse
            (
                "direct but rushed",
                "6 apples at $0.50 = $3.00, 4 oranges at $0.75 = $3.00. Total = $6.00."
            ),
            # y_3: arithmetic error — 4 * $0.75 computed as $2.50
            (
                "arithmetic error",
                "6 apples * $0.50 = $3.00\n"
                "4 oranges * $0.75 = $2.50\n"
                "Total = $3.00 + $2.50 = $5.50\n"
                "Maya spends $5.50."
            ),
        ],
    },
]

for p in reasoning_prompts:
    print(f"  {Fore.CYAN}── Reasoning Quality Demo ──{Style.RESET_ALL}")
    print(f"  Prompt: {p['user']}")
    print(f"  Expected answer: ${p['answer']}\n")

    responses = []
    rewards   = []

    for label, resp_text in p["rollouts"]:
        reward = score_math(resp_text, p["answer"])
        responses.append((label, resp_text, reward))
        rewards.append(reward)

    rewards_arr = np.array(rewards)
    advantages  = group_advantage(rewards_arr)

    for i, ((label, resp, reward), a) in enumerate(zip(responses, advantages)):
        print(f"  {Fore.YELLOW}y_{i+1} [{label}]{Style.RESET_ALL}")
        lines = [l.strip() for l in resp.replace("\r", "").split("\n") if l.strip()]
        for line in lines[:4]:
            print(f"    {line}")
        if len(lines) > 4:
            print(f"    ...")
        correct = reward == 1.0
        print(f"  Reward: {Fore.GREEN if correct else Fore.RED}{reward:.1f} ({'correct ✓' if correct else 'wrong ✗'}){Style.RESET_ALL}")
        print(f"  Group advantage Â: {a:+.4f}  →  {'increase prob ↑' if a > 0.001 else ('decrease prob ↓' if a < -0.001 else 'no change (all tied) →')}")
        print()

    print(f"  {Fore.GREEN}PPO would additionally compute:{Style.RESET_ALL}")
    print(f"    V(s_t) for every token in every response")
    print(f"    GAE advantage at each token: A_t = Σ (γλ)^l · δ_{{t+l}}")
    print(f"    Backward pass through value model to update V_φ")
    print()
    print(f"  {Fore.GREEN}GRPO computes:{Style.RESET_ALL}")
    print(f"    mean([{', '.join(f'{r:.1f}' for r in rewards_arr)}]) = {np.mean(rewards_arr):.3f}")
    print(f"    Â = ({', '.join(f'{a:.3f}' for a in advantages)})")
    print(f"    Done. No value model. ✓")
    print()

    print(f"  Paper result: on MATH benchmark, GRPO (52.1%) matches PPO (51.2%)")
    print(f"  with zero value model parameters and simpler training loop. ✓")

print(f"""
  Alignment Arc — complete:
    W13 InstructGPT: SFT + RM + PPO + value model (full RLHF)
    W14 DPO:         SFT + preference loss          (RM eliminated)
    W15 GRPO:        outcome rewards + group mean    (value model eliminated)

  Each paper stripped one component the previous treated as essential.
  What remains: SFT + a reward signal + group sampling.
""")

banner("grpo_finetune.py COMPLETE", Fore.YELLOW)