"""
rlhf_math.py — Week 13 · InstructGPT: Training Language Models to Follow Instructions
#ResearchRebuilt | Ouyang, Wu et al., NeurIPS 2022 | arxiv: 2203.02155

Covers the core RLHF pipeline from the paper — pure numpy, no API key needed.
  Section 1: The alignment problem — why bigger ≠ better-aligned
  Section 2: The 3-step pipeline — SFT → RM → PPO
  Section 3: Reward model math — Bradley-Terry preference model
  Section 4: PPO objective with KL penalty — the alignment tax tradeoff
  Section 5: Key results — win rates, hallucination, toxicity, alignment arc opener
"""

import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


def section(title: str, color=Fore.CYAN) -> None:
    bar = "─" * 64
    print(f"\n{color}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1 — THE ALIGNMENT PROBLEM (Paper §1)
# ─────────────────────────────────────────────────────────────
section("§1  The Alignment Problem — Why Bigger ≠ Better-Aligned", Fore.YELLOW)

"""
Paper (§1):
  GPT-3 (Brown et al. 2020) is trained to predict the next token.
  That objective is NOT the same as "be helpful to users."

  Consequence: larger models are NOT automatically more aligned.
    - They hallucinate confidently
    - They follow instructions literally, not intentionally
    - They produce toxic/harmful outputs even without intent

  The InstructGPT insight:
    Fine-tuning with human feedback aligns the model with user intent.
    A 1.3B InstructGPT is preferred over 175B GPT-3 — 100× fewer params, better alignment.

  The 3-step recipe:
    Step 1: SFT  — supervised fine-tune on labeler demonstrations
    Step 2: RM   — train a reward model on human preference rankings
    Step 3: PPO  — RL fine-tune using the reward model as the reward signal
"""

# The key result that opens the paper
print("  The paper's central claim (Figure 1 / Section 4.1):")
print()

win_rates = [
    ("1.3B InstructGPT vs 175B GPT-3",    "> 50%",  "smaller model wins — 100× fewer params"),
    ("175B InstructGPT vs 175B GPT-3",    "85% ± 3%", "same architecture, alignment changes everything"),
    ("175B InstructGPT vs few-shot GPT-3", "71% ± 4%", "even prompted GPT-3 loses"),
    ("175B InstructGPT vs SFT baseline",   "73.4% ± 2%", "RL beats supervised alone"),
    ("175B InstructGPT vs FLAN",           "78% ± 4%", "beats instruction-tuned FLAN"),
    ("175B InstructGPT vs T0",             "79% ± 4%", "beats instruction-tuned T0"),
]

print(f"  {'Comparison':<45} {'InstructGPT win rate':>22} {'Note'}")
print(f"  {'─'*90}")
for match, rate, note in win_rates:
    star = "85%" in rate or "1.3B" in match
    color = Fore.GREEN if star else ""
    reset = Style.RESET_ALL if star else ""
    print(f"  {color}{match:<45} {rate:>22}  {note}{reset}")

print(f"\n  {Fore.GREEN}Key: 1.3B InstructGPT beats 175B GPT-3.")
print(f"  Alignment > scale. This is the paper's thesis. ✓{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — THE 3-STEP PIPELINE (Paper §3)
# ─────────────────────────────────────────────────────────────
section("§3  The RLHF Pipeline — SFT → RM → PPO", Fore.MAGENTA)

"""
Paper §3:
  Three distinct training phases, each with its own dataset and objective.

  Step 1 — Supervised Fine-Tuning (SFT):
    Dataset: ~13K prompts (from API + labeler-written)
    Labels:  labeler-written ideal responses
    Train:   standard cross-entropy, 16 epochs, cosine decay
    Result:  a well-behaved base for the RM and PPO steps

  Step 2 — Reward Model (RM):
    Dataset: ~33K prompts, each with K=4–9 model outputs ranked by labelers
    Architecture: 6B GPT-3 with final token's embedding → scalar reward
    Train:  pairwise ranking loss (Bradley-Terry model)
    Result: a model that scores "how good" any response is

  Step 3 — PPO Fine-Tuning:
    Dataset: ~31K API prompts (no labels — reward comes from RM)
    Algorithm: PPO (Proximal Policy Optimization, Schulman et al. 2017)
    + KL penalty against SFT model (prevents reward hacking)
    + Optional: mix in pretraining gradient (PPO-ptx variant)
    Result: InstructGPT

  Labelers: ~40 contractors, with consistency checks
    Labeler-labeler agreement: 72.6% ± 1.5%
    Held-out labeler agreement: 77.3% ± 1.3%
    (comparable to researcher-researcher: 73% ± 4% in Stiennon et al.)
"""

pipeline = [
    ("Step 1", "SFT",  "~13K prompts", "labeler demos",  "cross-entropy",    "SFT model (1.3B/6B/175B)"),
    ("Step 2", "RM",   "~33K prompts", "ranked pairs (K=4–9)", "pairwise ranking", "reward model (6B)"),
    ("Step 3", "PPO",  "~31K prompts", "RM reward signal","PPO + KL penalty", "InstructGPT"),
]

print(f"  {'Step':<7} {'Phase':<6} {'Dataset':>14} {'Supervision':>22} {'Objective':>20} {'Output'}")
print(f"  {'─'*90}")
for step, phase, data, sup, obj, out in pipeline:
    color = Fore.GREEN if step == "Step 3" else ""
    reset = Style.RESET_ALL if color else ""
    print(f"  {color}{step:<7} {phase:<6} {data:>14} {sup:>22} {obj:>20}  {out}{reset}")

print(f"""
  Compute (from paper):
    175B SFT training  :  4.9 petaflops/s-days
    175B PPO-ptx       : 60.0 petaflops/s-days
    GPT-3 pre-training : 3,640 petaflops/s-days
    ─────────────────────────────────────────
    InstructGPT total  : ~1.8% of GPT-3 training compute
    {Fore.GREEN}↳ Alignment costs 1.8% of what it took to train the base model ✓{Style.RESET_ALL}
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — REWARD MODEL MATH (Paper §3.2)
# ─────────────────────────────────────────────────────────────
section("§3.2  Reward Model — Bradley-Terry Preference Loss", Fore.BLUE)

"""
Paper §3.2 (Equation 1):
  Given a prompt x and two responses y_w (winner) and y_l (loser),
  the reward model loss is:

    L_RM = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

  This is the Bradley-Terry model for pairwise preferences.
  σ(·) is the sigmoid function.
  r_θ(x, y) is the scalar reward output of the RM.

  Key: the RM never sees absolute "good" / "bad" labels.
  It only sees *relative* preferences — "A is better than B."
  This makes labeling much cheaper: ranking K outputs costs less than
  writing K good responses from scratch.

  For K outputs per prompt, you get K(K-1)/2 pairs.
  With K=4: 6 pairs per prompt.
  With K=9: 36 pairs per prompt.
  The paper uses K=4 to 9 → the 33K prompts generate ~100K+ training pairs.
"""

np.random.seed(42)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bradley_terry_loss(r_winner, r_loser):
    """Bradley-Terry pairwise ranking loss."""
    return -np.log(sigmoid(r_winner - r_loser))

# Simulate reward model outputs for 5 (prompt, response) pairs
print("  Bradley-Terry loss: L = -log σ(r_winner - r_loser)")
print()
print(f"  {'Scenario':<30} {'r_winner':>10} {'r_loser':>10} {'Δr':>8} {'Loss':>10} {'σ(Δr)':>10}")
print(f"  {'─'*80}")

scenarios = [
    ("Clear winner (trained)",     2.1, -0.8),
    ("Slight preference",          0.9,  0.5),
    ("Ambiguous (Δr ≈ 0)",         0.3,  0.2),
    ("Wrong direction (bad RM)",  -0.5,  1.2),
    ("Very confident winner",      3.5, -2.1),
]

for name, r_w, r_l in scenarios:
    loss = bradley_terry_loss(r_w, r_l)
    delta = r_w - r_l
    prob = sigmoid(delta)
    good = delta > 0
    color = Fore.GREEN if good else Fore.RED
    reset = Style.RESET_ALL
    print(f"  {color}{name:<30} {r_w:>10.2f} {r_l:>10.2f} {delta:>8.2f} {loss:>10.4f} {prob:>10.4f}{reset}")

print(f"""
  Observations:
    • When Δr is large and positive → loss ≈ 0 (model is confident and correct)
    • When Δr ≈ 0 → loss ≈ log(2) = 0.693 (coin flip, maximum uncertainty)
    • When Δr < 0 → high loss (model scores loser higher than winner)
    • The RM learns to assign higher scalars to preferred responses

  Paper's RM accuracy (Table 4):
    Training labeler agreement  : 72.4% ± 0.4%
    Held-out labeler agreement  : 69.6% ± 0.9%
    {Fore.GREEN}↳ RM learns human preferences at ~70% accuracy — close to human-human agreement ✓{Style.RESET_ALL}
""")

# Pairs from K outputs
print("  Pairs generated per prompt (K(K-1)/2):")
for K in [4, 5, 6, 9]:
    pairs = K * (K - 1) // 2
    print(f"    K={K} outputs → {pairs:>3} training pairs per prompt")
print(f"  {Fore.YELLOW}  ↳ 33K prompts × avg ~15 pairs ≈ ~100K+ comparisons for RM training ✓{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — PPO OBJECTIVE WITH KL PENALTY (Paper §3.3)
# ─────────────────────────────────────────────────────────────
section("§3.3  PPO Objective — Reward Maximisation + KL Penalty", Fore.CYAN)

"""
Paper §3.3 (Equation 2):
  The full PPO objective:

    objective(φ) = E[r_θ(x,y) - β · log(π_φ(y|x) / π_SFT(y|x))] + γ · E[log π_φ(x)]

  Three terms:
    1. r_θ(x,y)                              — RM reward (maximize this)
    2. -β · log(π_φ / π_SFT)                 — KL penalty (stay close to SFT)
    3. +γ · log π_φ(x)                        — pretraining log-likelihood (PPO-ptx only)

  Term 2 is critical: without the KL penalty, the model would "reward hack" —
  find adversarial outputs that score high on the RM but are not actually good.
  β controls the strength of the penalty.

  Term 3 (PPO-ptx variant): mixing in pretraining gradient prevents
  "alignment tax" — regressions on NLP benchmarks like SQuAD, DROP, HellaSwag.

  The alignment tax:
    Pure PPO: regressions on SQuAD, DROP, HellaSwag, WMT translation
    PPO-ptx:  regressions reduced; surpasses GPT-3 on HellaSwag
"""

# Simulate PPO objective across training steps
np.random.seed(7)
steps = np.arange(1, 101)

# Simulated RM reward increasing during training
rm_reward = 1.2 * (1 - np.exp(-steps / 30)) + np.random.normal(0, 0.05, len(steps))

# KL divergence grows if unconstrained
kl_unconstrained = 0.8 * (1 - np.exp(-steps / 20)) + np.random.normal(0, 0.02, len(steps))

# With β=0.02 KL penalty: KL stays bounded
beta = 0.02
kl_penalized = 0.3 * (1 - np.exp(-steps / 40)) + np.random.normal(0, 0.02, len(steps))

# Net objective: reward - beta*KL
obj_no_penalty = rm_reward - 0.0 * kl_unconstrained
obj_with_penalty = rm_reward - beta * kl_penalized * 50  # scaled for display

print("  PPO Objective = r_θ(x,y) − β·KL(π_φ ‖ π_SFT) [+ γ·pretraining]")
print()
print("  Snapshot at training milestones:")
print(f"  {'Step':>6} {'RM Reward':>12} {'KL (no pen.)':>14} {'KL (β·pen.)':>14} {'Net Obj (pen.)':>16}")
print(f"  {'─'*66}")

for i in [9, 24, 49, 74, 99]:
    kl_pen_val = beta * kl_penalized[i] * 50
    net = rm_reward[i] - kl_pen_val
    print(f"  {steps[i]:>6} {rm_reward[i]:>12.4f} {kl_unconstrained[i]:>14.4f} {kl_pen_val:>14.4f} {net:>16.4f}")

print(f"""
  Key insight — reward hacking without KL penalty:
    Without β: model maximises r_θ blindly → finds adversarial outputs
    With β:    model stays close to π_SFT → genuine quality improvement

  PPO-ptx variant (γ > 0):
    Mixes in pretraining log-likelihood gradient during PPO training
    Cost: slightly lower preference win rates
    Benefit: alignment tax on NLP benchmarks nearly eliminated
    {Fore.GREEN}↳ PPO-ptx surpasses GPT-3 on HellaSwag while remaining aligned ✓{Style.RESET_ALL}
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — KEY RESULTS DASHBOARD (Paper §4)
# ─────────────────────────────────────────────────────────────
section("§4  Results Dashboard — Alignment Actually Works", Fore.WHITE)

print("  ① Human preference win rates (vs 175B GPT-3):")
results_pref = [
    ("InstructGPT 175B (PPO-ptx)", "85% ± 3%", "★"),
    ("InstructGPT 175B (SFT only)", "~70%",     ""),
    ("InstructGPT 1.3B (PPO-ptx)", "> 50%",     "★★"),
    ("FLAN (175B)",                 "22% ± 4%",  ""),
    ("T0 (175B)",                   "21% ± 4%",  ""),
    ("GPT-3 175B (zero-shot)",      "15% ± 3%",  ""),
]

print(f"  {'Model':<35} {'Win rate vs GPT-3':>20} {'Note'}")
print(f"  {'─'*70}")
for model, rate, star in results_pref:
    color = Fore.GREEN if star else ""
    reset = Style.RESET_ALL if color else ""
    note = "100× fewer params — still wins" if "1.3B" in model else ("best" if "★" in star else "")
    print(f"  {color}{model:<35} {rate:>20}  {note}{reset}")

print(f"\n  ② Hallucination on closed-domain tasks (Figure 6):")
print(f"    GPT-3 175B     : 41% hallucination rate")
print(f"    InstructGPT    : 21% hallucination rate")
print(Fore.GREEN + f"    ↳ Hallucination cut by 49% ✓")

print(f"\n  ③ Toxicity — RealToxicityPrompts (§4.3):")
print(f"    With respectful prompt prefix:")
print(f"    GPT-3 (baseline) : higher toxicity")
print(f"    InstructGPT      : ~25% fewer toxic outputs")
print(Fore.GREEN + f"    ↳ Meaningful toxicity reduction without explicit toxicity training ✓")

print(f"\n  ④ Labeler agreement stats (Appendix B):")
print(f"    Training labelers ↔ training labelers  : 72.6% ± 1.5%")
print(f"    Held-out labelers ↔ held-out labelers  : 77.3% ± 1.3%")
print(f"    Researcher↔researcher (Stiennon 2020)  : 73% ± 4%")
print(Fore.YELLOW + f"    ↳ Human preference is noisy — ~73% agreement is the ceiling ✓")

print(f"""
  ⑤ Alignment Impact:
    InstructGPT demonstrated that aligning model behavior with human preferences
    via SFT + Bradley-Terry RM + PPO with KL penalty outperforms raw model scaling.
""")

section("rlhf_math.py COMPLETE — run rlhf_finetune.py next", Fore.YELLOW)