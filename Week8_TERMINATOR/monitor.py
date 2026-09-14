# monitor.py
# Display utilities for Week 8 TERMINATOR experiment.

from colorama import Fore, Style


def print_question_header(i: int, tc: dict, div: str):
    """Print a formatted question header."""
    diff_color = {
        "easy":   Fore.GREEN,
        "medium": Fore.YELLOW,
        "hard":   Fore.RED,
    }.get(tc.get("difficulty", ""), Fore.WHITE)

    print(f"\n{Fore.CYAN}{div}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Q{i}  {tc['label'].upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{div}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Question  :{Style.RESET_ALL} {tc['question'].replace(chr(10), ' ').strip()[:100]}")
    print(f"  {Fore.WHITE}Expected  :{Style.RESET_ALL} {tc['expected']}")
    print(
        f"  {Fore.WHITE}Difficulty:{Style.RESET_ALL} "
        f"{diff_color}{tc.get('difficulty', 'unknown').upper()}"
        f"{Style.RESET_ALL}\n"
    )


def print_exit_bar(exit_pos: int, full_words: int, width: int = 50):
    """
    Render a visual bar showing the TERMINATOR exit point within the full
    reasoning trace — makes the 'wasted compute' immediately legible.

    Example (exit at 40%, full=500 words):
      [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
      ↑ exit                                           ↑ end
    """
    if full_words == 0:
        return

    exit_frac = min(exit_pos / full_words, 1.0)
    filled    = int(exit_frac * width)
    empty     = width - filled

    bar = (
        f"  {Fore.GREEN}{'█' * filled}"
        f"{Fore.RED}{'░' * empty}"
        f"{Style.RESET_ALL}"
    )
    label = (
        f"  {Fore.GREEN}↑ exit ~{exit_pos}w{Style.RESET_ALL}"
        + " " * max(0, filled - len(f"↑ exit ~{exit_pos}w") + empty - 6)
        + f"{Fore.RED}↑ end {full_words}w{Style.RESET_ALL}"
    )

    print(f"\n  [{bar}]")
    print(label)