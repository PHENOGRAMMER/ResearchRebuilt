# benchmark.py
from memory_agents import BM25MemoryAgent, SemanticMemoryAgent, HybridMemoryAgent, MultiHopMemoryAgent
from evaluator import run_competency_test
from colorama import Fore, Style, init

init()

# ── COMPETENCY 1: Accurate Retrieval ──────────────────────────────────────────
# Inject facts across turns, retrieve specific ones later
ACCURATE_RETRIEVAL = [
    {"action": "add", "inject": "Alice works as a machine learning engineer at Google DeepMind in London."},
    {"action": "add", "inject": "Bob is a PhD student at MIT studying quantum computing."},
    {"action": "add", "inject": "Carol recently moved from New York to Singapore for a data science role."},
    {"action": "add", "inject": "David is the CTO of a startup building RAG systems in Berlin."},
    {"query": "Where does Alice work?", "expected": "Google DeepMind"},
    {"query": "What is Bob studying?", "expected": "quantum computing"},
    {"query": "Where did Carol move to?", "expected": "Singapore"},
    {"query": "What does David's company build?", "expected": "RAG"},
]

# ── COMPETENCY 2: Test-Time Learning ──────────────────────────────────────────
# Learn new facts mid-conversation without retraining
TEST_TIME_LEARNING = [
    {"action": "add", "inject": "The project deadline is March 15th."},
    {"query": "When is the project deadline?", "expected": "March 15"},
    {"action": "add", "inject": "The team uses Python and PyTorch for all ML experiments."},
    {"action": "add", "inject": "The model achieves 87.3% F1 score on the validation set."},
    {"query": "What F1 score does the model achieve?", "expected": "87.3"},
    {"action": "add", "inject": "After retraining, the model now achieves 91.2% F1 score."},
    {"query": "What is the current model F1 score?", "expected": "91.2"},
    {"query": "What frameworks does the team use?", "expected": "PyTorch"},
]

# ── COMPETENCY 3: Long-Range Understanding ─────────────────────────────────────
# Synthesize facts spread across many earlier turns
LONG_RANGE = [
    {"action": "add", "inject": "The company was founded in 2018 by Sarah Chen and James Park."},
    {"action": "add", "inject": "Sarah Chen has a PhD in Computer Science from Stanford University."},
    {"action": "add", "inject": "James Park previously worked at OpenAI for five years."},
    {"action": "add", "inject": "The company raised a $20M Series A in 2021."},
    {"action": "add", "inject": "Sarah Chen is currently serving as CEO."},
    {"action": "add", "inject": "James Park stepped down as CTO in 2023 to pursue academic research."},
    {"query": "Who founded the company and what are their backgrounds?", "expected": "Sarah"},
    {"query": "Is James Park still the CTO?", "expected": "no"},
    {"query": "What university did the CEO attend?", "expected": "Stanford"},
]

# ── COMPETENCY 4: Selective Forgetting ────────────────────────────────────────
# Update facts and verify old ones are not retrieved (the hardest competency)
SELECTIVE_FORGETTING = [
    {"action": "add", "inject": "The API endpoint is https://api.v1.example.com/predict"},
    {"query": "What is the API endpoint?", "expected": "v1"},
    # Fact changes — agent must forget old, learn new
    {"action": "update", "old_keyword": "API endpoint", "inject": "The API endpoint has been updated to https://api.v2.example.com/predict"},
    {"query": "What is the current API endpoint?", "expected": "v2"},
    {"query": "Is the v1 endpoint still in use?", "expected": "no"},
    # Another update
    {"action": "add", "inject": "The model serving version is v3.1."},
    {"action": "update", "old_keyword": "model serving version", "inject": "The model serving version has been upgraded to v4.0."},
    {"query": "What version is the model serving?", "expected": "v4.0"},
]

ALL_TESTS = [
    ("Accurate Retrieval",    ACCURATE_RETRIEVAL),
    ("Test-Time Learning",    TEST_TIME_LEARNING),
    ("Long-Range Understanding", LONG_RANGE),
    ("Selective Forgetting",  SELECTIVE_FORGETTING),
]

def run_benchmark():
    agents = {
        "BM25 Memory":     BM25MemoryAgent(),
        "Semantic Memory": SemanticMemoryAgent(),
        "Hybrid Memory":   HybridMemoryAgent(),
        "MultiHop Memory": MultiHopMemoryAgent(),
    }

    all_results = {name: [] for name in agents}

    for test_name, turns in ALL_TESTS:
        print(f"\n{Fore.CYAN}{'='*55}")
        print(f"COMPETENCY: {test_name}")
        print(f"{'='*55}{Style.RESET_ALL}")

        for agent_name, agent in agents.items():
            result = run_competency_test(agent, test_name, turns)
            all_results[agent_name].append(result)

            color = Fore.GREEN if result["accuracy"] >= 0.6 else Fore.RED
            print(f"\n{color}[{agent_name}] Accuracy: {result['accuracy']:.0%}{Style.RESET_ALL}")
            for r in result["results"]:
                mark = "✓" if r["correct"] else "✗"
                print(f"  {mark} Q: {r['query'][:50]}")
                print(f"    Expected: {r['expected']} | Got: {r['answer'][:60]}")

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'='*55}")
    print("RESULTS SUMMARY — Mini MemoryAgentBench Recreation")
    print(f"{'='*55}{Style.RESET_ALL}")
    print(f"{'Competency':<28} {'BM25':>8} {'Semantic':>10} {'Hybrid':>8} {'MultiHop':>10}")
    print("-" * 68)

    competencies = [t[0] for t in ALL_TESTS]
    for i, comp in enumerate(competencies):
        bm25_acc  = all_results["BM25 Memory"][i]["accuracy"]
        sem_acc   = all_results["Semantic Memory"][i]["accuracy"]
        hyb_acc   = all_results["Hybrid Memory"][i]["accuracy"]
        mhop_acc  = all_results["MultiHop Memory"][i]["accuracy"]
        print(f"{comp:<28} {bm25_acc:>7.0%}  {sem_acc:>9.0%} {hyb_acc:>7.0%}  {mhop_acc:>9.0%}")

    bm25_avg = sum(r["accuracy"] for r in all_results["BM25 Memory"]) / 4
    sem_avg  = sum(r["accuracy"] for r in all_results["Semantic Memory"]) / 4
    hyb_avg  = sum(r["accuracy"] for r in all_results["Hybrid Memory"]) / 4
    mhop_avg = sum(r["accuracy"] for r in all_results["MultiHop Memory"]) / 4
    print("-" * 68)
    print(f"{'Overall Average':<28} {bm25_avg:>7.0%}  {sem_avg:>9.0%} {hyb_avg:>7.0%}  {mhop_avg:>9.0%}")
    print(
    f"\n{Fore.YELLOW}"
    "Paper reference: The original paper reports that no evaluated system "
    "masters all four competencies and identifies selective forgetting as "
    "a particularly difficult capability."
    f"{Style.RESET_ALL}"
)

if __name__ == "__main__":
    run_benchmark()