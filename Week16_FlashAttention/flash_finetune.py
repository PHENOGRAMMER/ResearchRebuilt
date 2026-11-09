"""
flash_finetune.py — Week 16 · FlashAttention
#ResearchRebuilt | Dao, Fu, Ermon, Rudra, Ré · NeurIPS 2022 · arxiv: 2205.14135

Three experiments that test the paper's central argument:
  Experiment 1 — The argument test: memory-bound vs compute-bound operations
                 Does FLOP count predict speed? Or does memory movement?
  Experiment 2 — Tiling in action: online softmax vs materialised softmax
                 Same result, different memory footprint
  Experiment 3 — Context length scaling: what breaks first at long sequences

Model: openai/gpt-oss-120b via Groq API
"""

import os
import re
import time
import numpy as np
from groq import Groq
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "openai/gpt-oss-120b"


def strip_md(text: str) -> str:
    """Remove markdown bold/italic markers so output prints cleanly in terminal."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'__(.+?)__',     r'\1', text)
    text = re.sub(r'_(.+?)_',       r'\1', text)
    return text


def banner(text: str, color=Fore.CYAN) -> None:
    print(f"\nSTEP: {text}")


def call(system: str, user: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return strip_md(resp.choices[0].message.content.strip())


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — THE ARGUMENT TEST
# ─────────────────────────────────────────────────────────────
banner("1. Compare arithmetic intensity")

# Demonstrate memory-bound vs compute-bound numerically
print("RESULT: Arithmetic intensity")

np.random.seed(42)

# Two operations chosen to sit clearly on opposite sides of the ridge point.
#
# Op A: large linear projection (d_in=512 → d_out=512, batch=1024)
#   This is a weight-matrix-dominated matmul.
#   FLOPs = 2 * N * d_in * d_out  (each output element = d_in MACs)
#   Bytes = (N*d_in + d_in*d_out + N*d_out) * 2  (FP16 — input, weight, output)
#   Weight matrix dominates bytes: d_in*d_out >> N*d (for large d).
#   Intensity = FLOPs / Bytes >> 208 → compute-bound ✓
#
# Op B: softmax over N×N attention scores (N=1024)
#   FLOPs ≈ 5 per element (max, subtract, exp, sum, divide)
#   Bytes = 2 * N² * 2  (FP16 read + write of full score matrix)
#   Intensity ≈ 0.6 FLOP/byte → deeply memory-bound ✗

N    = 1024
d_in = 1024
d_out = 1024

flops_matmul   = 2 * N * d_in * d_out
bytes_matmul   = (N * d_in + d_in * d_out + N * d_out) * 2   # FP16
intensity_matmul = flops_matmul / bytes_matmul

flops_softmax  = 5 * N * N
bytes_softmax  = 2 * N * N * 2                                 # FP16 read+write
intensity_softmax = flops_softmax / bytes_softmax

a100_ridge = 208.0  # FLOP/byte

bound_a = "compute-bound ✓" if intensity_matmul > a100_ridge else "memory-bound  ✗"
bound_b = "compute-bound ✓" if intensity_softmax > a100_ridge else "memory-bound  ✗"
color_a = Fore.GREEN if intensity_matmul > a100_ridge else Fore.RED
color_b = Fore.GREEN if intensity_softmax > a100_ridge else Fore.RED

print(f"  Operation A: Large linear projection, N={N}, d_in={d_in}, d_out={d_out} (FP16)")
print(f"    FLOPs:                {flops_matmul/1e9:.2f}B")
print(f"    Bytes moved:          {bytes_matmul/1e6:.1f} MB")
print(f"    Arithmetic intensity: {intensity_matmul:.0f} FLOP/byte")
print(f"    Bottleneck: {color_a}{bound_a}{Style.RESET_ALL} (ridge = {a100_ridge:.0f} FLOP/byte)\n")

print(f"  Operation B: Softmax over attention scores S ∈ ℝ^{{{N}×{N}}} (FP16)")
print(f"    FLOPs:                {flops_softmax/1e6:.1f}M")
print(f"    Bytes moved:          {bytes_softmax/1e6:.1f} MB")
print(f"    Arithmetic intensity: {intensity_softmax:.1f} FLOP/byte")
print(f"    Bottleneck: {color_b}{bound_b}{Style.RESET_ALL} (ridge = {a100_ridge:.0f} FLOP/byte)\n")

print(f"  Intensity ratio (A/B): {intensity_matmul/intensity_softmax:.0f}×")
print(f"  Op A is {intensity_matmul/a100_ridge:.1f}× above the ridge → compute-bound")
print(f"  Op B is {a100_ridge/intensity_softmax:.0f}× below the ridge → memory-bound")
print("  FlashAttention eliminates Op B's HBM writes by keeping tiles in SRAM.")

# Ask the model to reason about this
sys_analyst = (
    "You are an expert in GPU architecture and deep learning systems. "
    "Answer precisely in 4-5 sentences. No LaTeX. No markdown."
)

q_argument = (
    f"On an A100 GPU the ridge point is ~{a100_ridge:.0f} FLOP/byte. "
    f"Operation A is a large linear projection: {flops_matmul/1e9:.2f}B FLOPs, {bytes_matmul/1e6:.0f} MB moved, "
    f"arithmetic intensity {intensity_matmul:.0f} FLOP/byte. "
    f"Operation B is softmax over an N×N attention score matrix (N={N}): {flops_softmax/1e6:.0f}M FLOPs, "
    f"{bytes_softmax/1e6:.0f} MB moved, arithmetic intensity {intensity_softmax:.1f} FLOP/byte. "
    "Which operation is the bottleneck and why? "
    "Then explain in 2 sentences what FlashAttention does to eliminate that bottleneck."
)

print("RESULT: Model reasoning")
print(f"  Q: {q_argument}\n")
answer = call(sys_analyst, q_argument)
for line in answer.split("\n"):
    if line.strip():
        print(f"  {line}")

print("\nRESULT: Paper Figure 2")
print(f"  GPT-2 medium attention on A100 (paper Figure 2):")
print(f"    Standard: 40.3 GB HBM traffic, 41.7 ms")
print(f"    FA:        4.4 GB HBM traffic,  7.3 ms")
print(f"    FLOPs:     identical — the speedup is 100% from IO reduction ✓")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — TILING IN ACTION
# ─────────────────────────────────────────────────────────────
banner("2. Verify online softmax and tiled attention")

# Run the numerical demo
np.random.seed(7)
N_seq  = 16
d_head = 4

Q = np.random.randn(N_seq, d_head) * 0.5
K = np.random.randn(N_seq, d_head) * 0.5
V = np.random.randn(N_seq, d_head) * 0.5
scale = 1.0 / np.sqrt(d_head)

# Standard attention: materialise full N×N score matrix
S_full = Q @ K.T * scale                            # shape: (N, N)
S_max  = S_full.max(axis=1, keepdims=True)
P_full = np.exp(S_full - S_max)
P_full = P_full / P_full.sum(axis=1, keepdims=True) # softmax
O_standard = P_full @ V                              # shape: (N, d)

# FlashAttention tiling: 4 blocks of 4 tokens each
block_size = 4
n_blocks   = N_seq // block_size

O_tiled = np.zeros((N_seq, d_head))
m_i = np.full((N_seq,), -np.inf)  # running row max
l_i = np.zeros((N_seq,))           # running sum of exp

for j in range(n_blocks):
    K_j = K[j*block_size:(j+1)*block_size]
    V_j = V[j*block_size:(j+1)*block_size]

    for i in range(n_blocks):
        Q_i = Q[i*block_size:(i+1)*block_size]
        rows = slice(i*block_size, (i+1)*block_size)

        S_ij = Q_i @ K_j.T * scale                 # (B_r, B_c) — stays in SRAM
        m_ij = S_ij.max(axis=1)                     # row max for this block

        m_new = np.maximum(m_i[rows], m_ij)
        l_new = (np.exp(m_i[rows] - m_new) * l_i[rows]
                 + np.exp(S_ij - m_new[:, None]).sum(axis=1))

        O_tiled[rows] = (
            (l_i[rows] * np.exp(m_i[rows] - m_new))[:, None] * O_tiled[rows]
            + np.exp(S_ij - m_new[:, None]) @ V_j
        ) / l_new[:, None]

        m_i[rows] = m_new
        l_i[rows] = l_new

max_err = np.max(np.abs(O_standard - O_tiled))

print(f"RESULT: Tiled output parity (N={N_seq}, d={d_head}, blocks={n_blocks})")
print(f"  Max absolute difference: {max_err:.2e}")
print(f"  Status: {'PASS' if max_err < 1e-10 else 'FAIL'}")

# Memory footprint comparison
print(f"RESULT: Activation memory (N={N_seq}, d={d_head}, FP32)")
elem_size = 4  # bytes, FP32
std_peak  = N_seq * N_seq * elem_size * 2  # S and P stored
fa_peak   = N_seq * d_head * elem_size + 2 * N_seq * elem_size  # O + (m, l)
print(f"    Standard: S + P matrices = {std_peak} bytes  ({N_seq}×{N_seq}×2 × 4)")
print(f"    FA tiled: O + stats      = {fa_peak} bytes   ({N_seq}×{d_head} + 2×{N_seq} × 4)")
print(f"    Saving:   {std_peak / fa_peak:.1f}× less activation memory at N={N_seq}")

N_large = 2048
d_large = 64
std_large = N_large * N_large * 4 * 2 / 1e6
fa_large  = (N_large * d_large * 4 + 2 * N_large * 4) / 1e6
print(f"\n  At production scale (N={N_large}, d={d_large}):")
print(f"    Standard: {std_large:.1f} MB per head")
print(f"    FA tiled:  {fa_large:.3f} MB per head")
print(f"    Saving:   {std_large/fa_large:.0f}× — this is what enables long-context training ✓\n")

# Ask model to explain the implication
sys_systems = (
    "You are an ML systems engineer. "
    "Answer in 3-4 sentences, plain text, no markdown, no LaTeX."
)
q_tiling = (
    "FlashAttention tiles the Q, K, V matrices into blocks and keeps computation "
    "in SRAM, never writing the N×N attention score matrix S to HBM. "
    "It uses online softmax with running (max, sum) statistics to produce exact output. "
    "In 3-4 sentences: why does this make long-context training feasible "
    "when standard attention runs out of memory?"
)
print("RESULT: Why tiling enables long context")
print(f"  Q: {q_tiling}\n")
answer2 = call(sys_systems, q_tiling)
for line in answer2.split("\n"):
    if line.strip():
        print(f"  {line}")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — CONTEXT LENGTH SCALING
# ─────────────────────────────────────────────────────────────
banner("3. Measure context-length memory scaling")

# Show the memory wall numerically
print("RESULT: Standard attention memory wall")
print(f"  Peak memory for attention score matrices (8 heads, d=64, FP16):")
print(f"  {'Seq length N':>14} {'S matrix (1 head)':>20} {'S + P (8 heads)':>18} {'Feasible?':>10}")
print(f"  {'─'*65}")

for N in [512, 1024, 2048, 4096, 8192, 16384]:
    s_one  = N * N * 2 / 1e6          # MB, FP16, 1 head
    sp_all = s_one * 2 * 8            # S+P, 8 heads
    ok = sp_all < 40 * 1024           # 40 GB A100
    color = Fore.GREEN if sp_all < 2000 else (Fore.YELLOW if sp_all < 10000 else Fore.RED)
    print(f"  {color}{N:>14} {s_one:>19.1f} MB {sp_all:>17.1f} MB {'✓' if ok else '✗ OOM':>10}{Style.RESET_ALL}")

print(f"\nRESULT: FlashAttention activation memory (O(Nd), all heads, FP16)")
d_fa = 64
for N in [512, 1024, 2048, 4096, 8192, 16384]:
    fa_mem = (N * d_fa * 8 * 2) / 1e6  # output only, 8 heads
    print(f"  {Fore.GREEN}  N={N:>5}: {fa_mem:.1f} MB{Style.RESET_ALL}")

# Ask model what longer context actually enables
sys_context = (
    "You are an NLP researcher familiar with transformer language models. "
    "Be specific and concrete. Plain text, no markdown, no LaTeX. 4-5 sentences."
)
q_context = (
    "Before FlashAttention, GPT-2 was trained with a maximum context of 1024 tokens. "
    "With FlashAttention, the same model architecture can be trained at 4096 tokens. "
    "What specific types of reasoning or tasks become possible or significantly better "
    "with 4× longer context? Give two or three concrete examples where the extra context "
    "makes a qualitative difference, not just a quantitative one."
)
print("\nRESULT: Longer-context reasoning")
print(f"  Q: {q_context}\n")
answer3 = call(sys_context, q_context, temperature=0.5)
for line in answer3.split("\n"):
    if line.strip():
        print(f"  {line}")

print("\nRESULT: Paper Table 4")
print(f"  GPT-2 at 1K context: perplexity 18.2")
print(f"  GPT-2 at 4K context: perplexity 17.5  (-0.7, trained 30% faster)")
print(f"  → Lower perplexity = better language model, at no extra compute cost.")
print(f"  → The only reason 4K wasn't used before: memory, not FLOPs. ✓")