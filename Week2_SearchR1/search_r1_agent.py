from groq import Groq, RateLimitError
from search_engine import search
from colorama import Fore, Style, init
import re, os, time, sys
from dotenv import load_dotenv

load_dotenv()
init()
sys.stdout.reconfigure(encoding="utf-8")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt — mirrors the structured rollout described in the paper.
# The key insight from Search-R1 is that the model is trained (via GRPO) to
# *decide when to search* and *what to search for*. We simulate that decision
# boundary with strict XML tags so a non-fine-tuned LLM can approximate it.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a reasoning agent that answers questions by searching Wikipedia.

You MUST follow this format on every turn, no exceptions:

<think>
Reason step by step. Identify exactly which fact you still need. Be specific.
</think>
<search>your search query here</search>

After you receive <information> back, continue:

<think>
Analyze what you retrieved. Do you now have ALL the facts needed to answer?
If yes, synthesize and write your final answer.
If no, search again for the missing fact.
</think>
<search>next targeted query</search>

When you have ALL facts from search results:

<think>
Combine the retrieved facts into a complete answer.
</think>
<answer>
Your concise final answer, citing only facts retrieved from search.
</answer>

Rules:
1. You MUST search at least once before giving an <answer>.
2. For multi-hop questions (A→B→C), search for each hop separately.
3. Never rely on your own memory — only facts from <information> blocks count.
4. Keep search queries short and specific (3–6 words). Avoid full sentences.
5. If a search returns irrelevant results, try a different, more specific query."""


# ---------------------------------------------------------------------------
# Tag extraction helpers
# ---------------------------------------------------------------------------

def extract_tag(text: str, tag: str) -> str | None:
    """Extract first occurrence of <tag>...</tag>. Returns stripped content or None."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_all_tags(text: str, tag: str) -> list[str]:
    """Extract all occurrences of <tag>...</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]

# ---------------------------------------------------------------------------
# Evidence Sufficiency Checker
# ---------------------------------------------------------------------------

def evidence_is_sufficient(messages: list[dict]) -> bool:
    """
    Ask the LLM whether the accumulated <information> blocks
    are sufficient to answer the original question.

    Returns True if enough evidence has been gathered.
    """

    prompt = (
        "Review all retrieved information.\n\n"
        "Compare the retrieved evidence against the ORIGINAL QUESTION.\n"
        "Only answer YES if you can answer the ORIGINAL QUESTION completely.\n"
        "If any reasoning hop is still missing, answer NO.\n"
        "Reply ONLY with YES or NO."
    )

    msgs = messages + [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs,
            temperature=0,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().upper()

        return answer.startswith("YES")

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fallback synthesis — called when max_turns is hit without a final <answer>
# Asks the model to synthesize from everything it has gathered so far.
# This prevents the unhelpful "Max turns reached" blank return.
# ---------------------------------------------------------------------------

def _force_final_answer(messages: list[dict], query: str) -> str:
    """Inject a synthesis prompt and extract whatever answer the model produces."""
    synthesis_prompt = (
        "You have now done several searches. Based ONLY on the <information> blocks "
        "you have received so far, give your best final answer to the original question. "
        "Use this format:\n"
        "<think>\nSynthesize what you know.\n</think>\n"
        "<answer>\nYour best answer here.\n</answer>"
    )
    messages_copy = messages + [{"role": "user", "content": synthesis_prompt}]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages_copy,
                temperature=0.2,
                max_tokens=400,
                # No stop token — we want the full answer block
            )
            out = resp.choices[0].message.content.strip()
            answer = extract_tag(out, "answer")
            if answer:
                return answer.strip()
            clean = out
            clean = re.sub(
                r"<think>.*?</think>",
                "",
                clean,
                flags=re.DOTALL,
            )
            clean = re.sub(
                r"<think>.*",
                "",
                clean,
                flags=re.DOTALL,
            )
            clean = re.sub(
                r"</?answer>",
                "",
                clean,
            )
            clean = clean.strip()
            return clean
        except RateLimitError:
            time.sleep(10)

    return "Could not synthesize an answer from retrieved context."


# ---------------------------------------------------------------------------
# Main Search-R1 inference loop
# ---------------------------------------------------------------------------

def run_search_r1(query: str, max_turns: int = 8) -> dict:
    """
    Simulate the Search-R1 inference loop.

    Each turn:
      1. LLM generates until </search> stop token (or gives <answer>)
      2. We extract the search query, retrieve from Wikipedia
      3. Inject <information>...</information> and continue

    Returns:
        dict with keys: query, answer, search_count, trace
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print("Search-R1 Inference Loop")
    print(f"Query: {query}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                "Start with <think> to identify what to search for first, "
                "then issue a <search> query."
            ),
        },
    ]

    search_count = 0
    previous_searches = set()
    full_trace = f"Query: {query}\n"
    accumulated_info: list[str] = []  # track all retrieved snippets for fallback

    for turn in range(max_turns):
        # Brief pause to avoid Groq rate limits on free tier
        time.sleep(2)

        # ── LLM call — stop at </search> so we can inject retrieval ──────────
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                    stop=["</search>"],
                )
                break
            except RateLimitError as e:
                wait = 15 * (attempt + 1)
                print(
                    f"{Fore.RED}[Rate limit — waiting {wait}s]{Style.RESET_ALL}"
                )
                time.sleep(wait)
        else:
            return {
                "query": query,
                "answer": "Rate limit exceeded after retries.",
                "search_count": search_count,
                "trace": full_trace,
            }

        llm_output = response.choices[0].message.content.strip()

        # ── The model may open <search> without closing it (stop cut it off) ──
        # Reconstruct the closing tag so our regex can find the query.
        if "<search>" in llm_output and "</search>" not in llm_output:
            llm_output += "</search>"

        # ── Extract structured components ─────────────────────────────────────
        think_content = extract_tag(llm_output, "think")
        answer = extract_tag(llm_output, "answer")
        search_queries = extract_all_tags(llm_output, "search")

        # Print the think block (truncated for readability)
        if think_content:
            print(f"{Fore.YELLOW}[THINK — Turn {turn + 1}]{Style.RESET_ALL}")
            preview = think_content[:250]
            if len(think_content) > 250:
                preview += "..."
            print(f"  {preview}")

        # ── Final answer found — return immediately ───────────────────────────
        if answer:
            print(f"\n{Fore.MAGENTA}[FINAL ANSWER]{Style.RESET_ALL}")
            print(f"  {answer}")
            full_trace += f"\n[ANSWER]: {answer}"
            return {
                "query": query,
                "answer": answer,
                "search_count": search_count,
                "trace": full_trace,
            }

        # ── Search query found — retrieve and inject ──────────────────────────
        if search_queries:
            sq = search_queries[0].strip()
            normalized = sq.lower().strip()
            if normalized in previous_searches:
                print(
                    f"{Fore.RED}[Duplicate Search skipped]{Style.RESET_ALL}"
                )

                messages.append(
                    {
                        "role": "user",
                        "content": 
                            "That search has already been performed. "
                            "Try a different search query."
                    }
                )

                continue
            previous_searches.add(normalized)


            # Guard: model sometimes emits "None needed" or similar
            non_search_patterns = [
                r"^none",
                r"^no search",
                r"^no query",
                r"^query not",
                r"^i have",
                r"^enough",
                r"^done",
                r"^finished",
                r"^not needed",
                r"^n/a",
            ]
            is_fake_search = any(
                re.match(p, sq.lower()) for p in non_search_patterns
            )

            if is_fake_search:
                # Model thinks it's done but hasn't emitted <answer> — nudge it
                print(
                    f"{Fore.RED}[Fake search detected: '{sq}' — nudging for answer]{Style.RESET_ALL}"
                )
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "You indicated no further search is needed. "
                        "Please now write your final answer using <answer> tags."
                    ),
                })
                continue

            if search_count >= 2 and evidence_is_sufficient(messages):
                print(
                    f"{Fore.GREEN}[Evidence sufficient — stopping search]{Style.RESET_ALL}"
                )

                final_answer = _force_final_answer(messages, query)

                print(f"\n{Fore.MAGENTA}[FINAL ANSWER]{Style.RESET_ALL}")
                print(f"  {final_answer}")

                return {
                    "query": query,
                    "answer": final_answer,
                    "search_count": search_count,
                    "trace": full_trace,
                }
            search_count += 1
            print(f"\n{Fore.GREEN}[SEARCH #{search_count}]{Style.RESET_ALL}")
            print(f"  Query: {sq}")

            retrieved = search(sq)
            snippet = retrieved[:200]
            print(f"  Retrieved: {snippet}...")

            accumulated_info.append(f"[Search #{search_count}: '{sq}']\n{retrieved}")
            full_trace += f"\n[SEARCH #{search_count}]: {sq}\n[INFO]: {retrieved[:120]}...\n"

            # Append assistant turn and the retrieved information as a user turn.
            # This is the core interleaving mechanic from the paper (Section 3).
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": (
                    f"<information>\n{retrieved}\n</information>\n\n"
                    "Continue your reasoning. Do you have all facts needed, "
                    "or do you need another search?"
                ),
            })

        else:
            # Model produced neither a search nor an answer — it's stuck
            print(f"{Fore.RED}[Stuck — no <search> or <answer> tag found]{Style.RESET_ALL}")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": (
                    "You must either issue a <search> query to retrieve more information, "
                    "or write your final <answer> if you already have all the facts."
                ),
            })

    # ── Max turns reached — attempt forced synthesis rather than blank return ──
    print(f"\n{Fore.RED}[Max turns reached — attempting synthesis from {len(accumulated_info)} retrieved snippets]{Style.RESET_ALL}")
    synthesized = _force_final_answer(messages, query)
    print(f"{Fore.MAGENTA}[SYNTHESIZED ANSWER]{Style.RESET_ALL}")
    print(f"  {synthesized}")

    return {
        "query": query,
        "answer": f"[Synthesized] {synthesized}",
        "search_count": search_count,
        "trace": full_trace,
    }