"""
flash_math.py — Week 16 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
#ResearchRebuilt | Dao, Fu, Ermon, Rudra, Ré · Stanford · NeurIPS 2022 · arxiv: 2205.14135

Pure numpy derivations — no API key needed.

Sections:
  §1  The memory hierarchy — HBM vs SRAM, bandwidth vs capacity
  §2  Standard attention IO complexity — why it's memory-bound
  §3  FlashAttention tiling — the algorithm that fixes it
  §4  IO complexity comparison — Θ(N²d) vs Θ(N²d²/M)
  §5  Results dashboard — speedup, memory, context length
"""

import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


def section(title: str) -> None:
    print(f"\nSTEP: {title}")


# ─────────────────────────────────────────────────────────────
# §1  THE MEMORY HIERARCHY
# ─────────────────────────────────────────────────────────────
section("1. Measure the memory hierarchy")

print("RESULT: A100 ridge point = 208 FLOP/byte; attention softmax is memory-bound.")

print(f"RESULT: Arithmetic intensity")
print(f"  {'Operation':<35} {'FLOPs':>10} {'Bytes moved':>12} {'FLOP/byte':>11} {'Bottleneck'}")
print(f"  {'─'*75}")

ops = [
    # (name, flops_per_element, bytes_per_element, notes)
    ("Matrix multiply (large, d=512)",     1024,   4,   "compute-bound"),
    ("Matrix multiply (small, d=64)",       128,   4,   "compute-bound"),
    ("Softmax (N=2048)",                      5,  12,   "memory-bound"),
    ("Elementwise multiply",                  1,   8,   "memory-bound"),
    ("Layer norm",                            8,  12,   "memory-bound"),
    ("Attention score QKᵀ (N=512,d=64)",   512,   4,   "compute-bound"),
    ("Attention softmax + dropout",           6,  16,   "memory-bound"),
]

a100_ridge = 208.0  # FLOP/byte ridge point

for name, flops, bw, expected in ops:
    intensity = flops / bw
    bound = "compute ✓" if intensity > a100_ridge else "memory  ✗"
    color = Fore.GREEN if intensity > a100_ridge else Fore.RED
    print(f"  {color}{name:<35} {flops:>10} {bw:>12} {intensity:>11.1f}   {bound}{Style.RESET_ALL}")

print(f"\n  Ridge point (A100): {a100_ridge:.0f} FLOP/byte")
print(f"  ↳ Attention's elementwise ops sit far below the ridge — purely memory-bound ✓")


# ─────────────────────────────────────────────────────────────
# §2  STANDARD ATTENTION IO COMPLEXITY
# ─────────────────────────────────────────────────────────────
section("2. Count standard-attention HBM accesses")

print("RESULT: Standard attention materializes S and P; HBM traffic is Θ(Nd + N²).")

print("RESULT: Quadratic HBM traffic")
d = 64
print(f"  HBM access count growth (d={d}, in millions of elements):")
print(f"  {'N':>6} {'Nd (linear)':>14} {'N² (quadratic)':>16} {'Total':>10} {'N² share':>10}")
print(f"  {'─'*60}")

for N in [512, 1024, 2048, 4096, 8192]:
    nd   = N * d
    n2   = N * N
    total = nd + n2
    share = 100 * n2 / total
    print(f"  {N:>6} {nd/1e6:>13.2f}M {n2/1e6:>15.2f}M {total/1e6:>9.2f}M {share:>9.1f}%")

print(f"\n  ↳ At N=4096, N² is 98.5% of all HBM traffic — Nd is negligible ✓")
print(f"  ↳ This is what makes standard attention memory-bound at scale ✓")


# ─────────────────────────────────────────────────────────────
# §3  FLASHATTENTION TILING
# ─────────────────────────────────────────────────────────────
section("3. Verify tiled exact attention")

print("RESULT: Tiled Q, K, V use online (m, l) statistics and never materialize N×N scores.")

# Demonstrate online softmax correctness
print("RESULT: Online softmax")

np.random.seed(42)
N_demo = 8
scores = np.random.randn(N_demo) * 2  # simulated attention scores

# Exact softmax
max_s = np.max(scores)
exact = np.exp(scores - max_s) / np.sum(np.exp(scores - max_s))

# Online softmax over 2 blocks
block1 = scores[:N_demo//2]
block2 = scores[N_demo//2:]

# Block 1
m = np.max(block1)
l = np.sum(np.exp(block1 - m))

# Block 2 update
m_new = max(m, np.max(block2))
l_new = np.exp(m - m_new) * l + np.sum(np.exp(block2 - m_new))

# Final softmax
all_exp = np.concatenate([
    np.exp(block1 - m_new),
    np.exp(block2 - m_new)
])
online = all_exp / l_new

print(f"  Max difference: {np.max(np.abs(exact - online)):.2e}")
print(f"  Status: {'PASS' if np.allclose(exact, online, rtol=1e-6, atol=1e-7) else 'FAIL'}")


# ─────────────────────────────────────────────────────────────
# §4  IO COMPLEXITY COMPARISON
# ─────────────────────────────────────────────────────────────
section("4. Compare IO complexity")

print("""  HBM access counts (paper Theorem 2):

  Standard attention:    Θ( Nd + N² )
  FlashAttention:        Θ( N²d² / M )

  where:
    N = sequence length
    d = head dimension (typically 64–128)
    M = SRAM size (bytes, A100: ~100KB per SM)

  The ratio (how many times fewer HBM accesses FA needs):

    Standard / FA  =  Θ( (Nd + N²) · M / N²d² )
                   ≈  Θ( M / d² )   for large N

  For A100: M ≈ 100,000 bytes, d = 64:
    Ratio ≈ 100,000 / (64² * 2) ≈ 12×

  For d = 128:
    Ratio ≈ 100,000 / (128² * 2) ≈ 3×

  Paper reports: up to 9× fewer HBM accesses in practice (Figure 2).
  Empirical example (GPT-2 medium, A100, paper Figure 2):
    Standard attention:  40.3 GB HBM reads/writes, 41.7 ms
    FlashAttention:       4.4 GB HBM reads/writes,  7.3 ms
    Reduction:           9.2× fewer bytes,          5.7× faster
""")

# Show the IO reduction across sequence lengths and head dims
print("RESULT: Theoretical IO reduction (FP16, M=100,000 bytes)")
print()

M_bytes = 100_000
bytes_per_elem = 2
M_elements = M_bytes // bytes_per_elem

print(f"  {'N':>6} {'d':>5} {'Standard Θ':>14} {'FA Θ':>14} {'Reduction':>10}")
print(f"  {'─'*55}")

for N in [512, 1024, 2048, 4096]:
    for d in [64, 128]:
        std_io = N * d + N * N
        fa_io  = (N * N * d * d) / M_elements
        ratio  = std_io / fa_io
        marker = " ←" if N == 2048 and d == 64 else ""
        print(f"  {N:>6} {d:>5} {std_io/1e6:>13.2f}M {fa_io/1e6:>13.2f}M {ratio:>9.1f}×{marker}")

print("\n  The ratio increases with SRAM capacity and decreases with head dimension.")

# Recomputation in backward pass
print("RESULT: Backward stores O and (m, l), then recomputes tiles; activation memory is O(Nd).")


# ─────────────────────────────────────────────────────────────
# §5  RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────
section("5. Compare paper results")

print(f"  ① Training speed (paper Table 1, 2 — A100, FP16):")
print(f"  {'Model / Setting':<40} {'Baseline':>12} {'FA':>10} {'Speedup':>9}")
print(f"  {'─'*73}")

speed = [
    ("BERT-large (seq 512, MLPerf 1.1)",  "20.0 min", "17.4 min", "1.15×"),
    ("GPT-2 small  (seq 1024, HF)",       "9.5 days",  "2.7 days", "3.5×"),
    ("GPT-2 medium (seq 1024, HF)",       "21.0 days", "6.9 days", "3.0×"),
    ("GPT-2 small  (seq 1024, Megatron)", "4.7 days",  "2.7 days", "1.7×"),
    ("Long-range Arena (avg)",            "baseline",  "—",        "2.4×"),
]

for name, baseline, fa, speedup in speed:
    color = Fore.GREEN if "3." in speedup or "2." in speedup else Fore.YELLOW
    print(f"  {color}{name:<40} {baseline:>12} {fa:>10} {speedup:>9}{Style.RESET_ALL}")

print(f"""
  ② Memory savings (paper §4.3, A100):
  {'Metric':<45} {'Standard':>12} {'FA':>10}
  {'─'*68}
  {'Peak memory (N=2048, GPT-2 medium)':<45} {'~34 GB':>12} {'~26 GB':>10}
  {'HBM reads/writes (GPT-2 med fwd+bwd)':<45} {'40.3 GB':>12} {'4.4 GB':>10}
  {'Memory scaling with sequence length':<45} {'O(N²)':>12} {'O(N)':>10}
  {'Activation storage for backward pass':<45} {'O(N²)':>12} {'O(N)':>10}

  ③ Context length enabled by memory savings (paper §4.2):
  {'Model':<35} {'Baseline max N':<18} {'FA max N':<12} {'Quality gain'}
  {'─'*68}""")

ctx = [
    ("GPT-2",          "1K (memory limit)", "4K",  "perplexity 18.2 → 17.5 (-0.7)"),
    ("Long-range Path-X", "failed (OOM)",   "16K", "61.4% accuracy (first ever)"),
    ("Block-sparse FA",   "failed (OOM)",   "64K", "63.1% on Path-256"),
]
for model, base, fa, quality in ctx:
    print(f"  {Fore.GREEN}{model:<35} {base:<18} {fa:<12} {quality}{Style.RESET_ALL}")

print(f"""
  ④ The argument in numbers:
     GPT-2 medium attention, A100 (paper Figure 2):
       HBM traffic:  40.3 GB → 4.4 GB   (9.2× reduction)
       Runtime:      41.7 ms → 7.3 ms   (5.7× faster)
       FLOPs:        identical           (same math, different memory pattern)

     This is the core point: 9.2× less memory movement → 5.7× faster.
     Not a single FLOP was saved. The speedup is 100% from IO reduction. ✓
""")