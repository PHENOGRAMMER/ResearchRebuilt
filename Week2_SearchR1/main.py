"""
main.py

Compares three retrieval strategies on multi-hop factual questions:
  1. No Retrieval   — LLM answers from parametric memory only
  2. Naive RAG      — single Wikipedia retrieval, no reasoning loop
  3. Search-R1      — iterative think→search→inform loop (paper: arXiv:2503.09516)

Query design follows HotpotQA / 2WikiMultiHopQA / MuSiQue conventions used in
the paper's evaluation (Section 4): questions require chaining 2+ Wikipedia
facts that cannot be answered by a single retrieval.

Fix vs v1: queries are genuine multi-hop questions — the LLM cannot decompose
them by reading the query alone; it must search iteratively to bridge the hops.
"""

import sys
import os
import json
import glob
from colorama import Fore, Style, init

from search_r1_agent import run_search_r1
from comparator import run_naive_rag, run_no_retrieval
from search_engine import search

init()
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Test queries — genuine 2-hop and 3-hop questions
# Each question requires chaining at least two Wikipedia lookups.
# Ground-truth answers verified against Wikipedia (August 2025).
# ---------------------------------------------------------------------------
test_queries = [
    {
        "query": "Which university did the creator of Python graduate from?",
        "expected": "University of Amsterdam",
        "expected_key": "amsterdam",
        "short_name": "Python Creator",
        "hops": 2,
    },
    {
        "query": "What is the capital of the country where the Eiffel Tower is located?",
        "expected": "Paris",
        "expected_key": "paris",
        "short_name": "Eiffel Tower",
        "hops": 2,
    },
    {
        "query": "Who was the US president when the Berlin Wall fell?",
        "expected": "George H. W. Bush",
        "expected_key": "bush",
        "short_name": "Berlin Wall",
        "hops": 2,
    },
]


# ---------------------------------------------------------------------------
# Helper — extract a short naive RAG query from the question
# ---------------------------------------------------------------------------

def _naive_rag_query(question: str) -> str:
    """
    Simple keyword extractor for the single-retrieval baseline.
    Strips question words and short tokens; takes the first 6 content words.
    """
    skip = {"what", "where", "when", "who", "which", "whose", "that", "this",
            "the", "its", "was", "were", "did", "does", "have", "been",
            "name", "named", "also", "from", "most", "more", "only", "both"}
    tokens = [
        w.strip("?,.()'\"").lower()
        for w in question.split()
        if len(w.strip("?,.()'\"")) > 3
    ]
    keywords = [t for t in tokens if t not in skip][:6]
    return " ".join(keywords)


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def main():
    print(f"\n{Fore.CYAN}{'='*60}")
    print("Search-R1 vs Baselines — Comparison")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    results = []

    for tc in test_queries:
        query    = tc["query"]
        expected = tc["expected"]
        hops     = tc["hops"]

        print(f"\n{Fore.WHITE}{'─'*60}")
        print(f"QUERY [{hops}-hop]: {query}")
        print(f"Expected: {expected}")
        print(f"{'─'*60}{Style.RESET_ALL}\n")

        # ── Baseline 1: No retrieval ──────────────────────────────────────────
        no_ret = run_no_retrieval(query)
        print(f"{Fore.RED}[No Retrieval]: {no_ret[:120]}{Style.RESET_ALL}")

        # ── Baseline 2: Naive RAG (one retrieval, one generation) ─────────────
        naive_query   = _naive_rag_query(query)
        naive_context = search(naive_query)
        naive         = run_naive_rag(query, naive_context)
        print(f"{Fore.YELLOW}[Naive RAG]:    {naive[:120]}{Style.RESET_ALL}")

        # ── Search-R1: iterative think → search → inform loop ─────────────────
        r1_result = run_search_r1(query, max_turns=4)
        answer_preview = r1_result["answer"][:120]
        print(f"{Fore.GREEN}[Search-R1]:    {answer_preview}{Style.RESET_ALL}")
        print(f"  Searches used: {r1_result['search_count']}")

        results.append({
            "query":       query,
            "expected":    expected,
            "short_name":  tc["short_name"],
            "hops":        hops,
            "no_retrieval": no_ret,
            "naive_rag":   naive,
            "search_r1":   r1_result["answer"],
            "searches":    r1_result["search_count"],
            "correct_vanilla": tc["expected_key"].lower() in no_ret.lower(),
            "correct_naive":   tc["expected_key"].lower() in naive.lower(),
            "correct_r1":      tc["expected_key"].lower() in r1_result["answer"].lower(),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'='*60}")
    print("RESULTS")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    print("| Query | Vanilla | Naive RAG | Search-R1 | Searches | Correct |")
    print("|---|---|---|---|---|---|")
    for r in results:
        vanilla_icon = "✅" if r["correct_vanilla"] else "❌"
        naive_icon   = "✅" if r["correct_naive"] else "❌"
        r1_icon      = "✅" if r["correct_r1"] else "❌"
        
        print(f"| {r['short_name']} | {vanilla_icon} | {naive_icon} | {r1_icon} | {r['searches']} | {r1_icon} |")

    # ── Save Results ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    existing_runs = glob.glob("results/run_*.json")
    next_run_num = len(existing_runs) + 1
    run_filename = f"results/run_{next_run_num:03d}.json"
    
    with open(run_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved run results to {run_filename}")

if __name__ == "__main__":
    main()