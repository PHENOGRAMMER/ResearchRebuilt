# main.py
from memory_bank import MemoryBank
from naive_memory_manager import naive_manage
from rl_memory_manager import rl_manage
from answer_agent import answer_with_distillation
from colorama import Fore, Style, init

init()

# ── Scenario 1: Project Requirements (Evolving & Contradictory) ───────────────
DIALOGUE_SESSIONS = [
    "The new website will have a dark mode feature.",
    "The website will also support multiple languages.",
    "Actually, we decided to drop dark mode for the initial release.",
    "The supported languages will be English, Spanish, and French.",
]

QUESTIONS = [
    "Will the website have dark mode in the initial release?",
    "What languages will the website support?",
]

# ── Scenario 2: Medical Patient History (Complex Context) ───────────────────────
DIALOGUE_2 = [
    "Patient John reported mild headaches starting last Monday.",
    "John was prescribed Ibuprofen 400mg twice a day.",
    "John's headaches have worsened; he now experiences nausea.",
    "Switched John's medication from Ibuprofen to Sumatriptan 50mg.",
]

QUESTIONS_2 = [
    "What are John's current symptoms?",
    "What medication is John currently taking?",
]

def run_scenario(title, sessions, questions, label=""):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{title}")
    print(f"{'='*60}{Style.RESET_ALL}")

    naive_bank = MemoryBank()
    rl_bank = MemoryBank()

    print(f"\n{Fore.YELLOW}── Processing dialogue sessions ──{Style.RESET_ALL}")

    for i, turn in enumerate(sessions):
        print(f"\n  Turn {i+1}: \"{turn[:70]}...\"" if len(turn) > 70 else f"\n  Turn {i+1}: \"{turn}\"")

        # Naive manager
        naive_op = naive_manage(naive_bank, turn)
        print(f"  {Fore.RED}[Naive]   → {naive_op}{Style.RESET_ALL}")

        # RL-simulated manager
        rl_op, rl_reason = rl_manage(rl_bank, turn)
        print(f"  {Fore.GREEN}[RL-sim]  → {rl_op} | Reasoning: {rl_reason[:80]}{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}── Final Memory Banks ──{Style.RESET_ALL}")
    print(f"\n{Fore.RED}[Naive Memory Bank]{Style.RESET_ALL}")
    naive_bank.show()
    print(f"\n{Fore.GREEN}[RL-Simulated Memory Bank]{Style.RESET_ALL}")
    rl_bank.show()

    print(f"\n{Fore.YELLOW}── Answering Questions ──{Style.RESET_ALL}")
    for q in questions:
        print(f"\n  Q: {q}")
        naive_ans = answer_with_distillation(naive_bank, q)
        rl_ans = answer_with_distillation(rl_bank, q)
        print(f"  {Fore.RED}[Naive]  {naive_ans['answer'][:100]} "
              f"(retrieved {naive_ans['retrieved_count']} → distilled {naive_ans['distilled_count']}){Style.RESET_ALL}")
        print(f"  {Fore.GREEN}[RL-sim] {rl_ans['answer'][:100]} "
              f"(retrieved {rl_ans['retrieved_count']} → distilled {rl_ans['distilled_count']}){Style.RESET_ALL}")

if __name__ == "__main__":
    run_scenario(
        "SCENARIO 1: Project Requirements (Evolving & Contradictory)",
        DIALOGUE_SESSIONS, QUESTIONS
    )
    run_scenario(
        "SCENARIO 2: Medical Patient History (Complex Context)",
        DIALOGUE_2, QUESTIONS_2
    )