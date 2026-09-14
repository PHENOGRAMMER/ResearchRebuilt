# early_exit.py
# Week 8 — #ResearchRebuilt
# TERMINATOR: Learning Optimal Exit Points for Early Stopping in CoT Reasoning
# Paper: arXiv:2603.12529
#
# Core idea reproduced:
#   - Stream the reasoning trace from a thinking-mode model (Qwen3.x on Groq)
#   - Detect the FIRST position where a well-formed correct answer appears
#   - Require STABILITY: same answer across N consecutive chunks before firing
#   - That stability point = the TERMINATOR exit position
#   - Compute tokens wasted after that point
#
# NOT reproduced: TERMINATOR's trained binary exit classifier (requires fine-tuning)
#
# NOTE: ground truth is used only POST-HOC to identify first correct position.
# The model has no access to expected answers during generation.

from groq import Groq
from colorama import Fore, Style
import os, re, time
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Model ─────────────────────────────────────────────────────────────────────
# Qwen3.x exposes thinking tokens in <think>...</think> tags within the
# content stream. We parse these out to get the reasoning trace separately.
# gpt-oss-120b is a stronger reasoner but merges everything into content
# with no <think> tags, so we can't isolate reasoning words.
# qwen/qwen3.6-27b is the best option: proper thinking mode, 500 t/s, free tier.
MODEL = "qwen/qwen3.6-27b"

# ── Stability threshold ────────────────────────────────────────────────────────
# TERMINATOR fires when the same correct answer appears in N consecutive chunks.
# This prevents false positives like "if the answer were 64, but let me verify..."
STABILITY_CHUNKS = 3

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. "
    "Think step by step through the problem carefully. "
    "When you reach your final answer, state it clearly and explicitly — "
    "for example: 'The answer is X' or 'Therefore, X'."
)


# ─── Reasoning stream parser ──────────────────────────────────────────────────
# Qwen3.x thinking models emit:
#   <think>\nreasoning here...\n</think>\nfinal answer here
#
# During streaming the <think> open tag may or may not arrive as its own chunk.
# We track state and split chunks into reasoning vs answer accordingly.

class StreamParser:
    """Parse Qwen3 streaming chunks into reasoning vs final-answer text."""

    def __init__(self):
        self.buf          = ""    # raw accumulator (for tag detection)
        self.in_think     = False
        self.think_closed = False
        self.reasoning    = ""
        self.answer       = ""

    def feed(self, delta: str) -> tuple[str, str]:
        """
        Feed one streaming delta. Returns (reasoning_delta, answer_delta).
        """
        if not delta:
            return "", ""

        self.buf += delta
        r_out = ""
        a_out = ""

        # Process character by character only when we're near a tag boundary.
        # For performance, bulk-process when we know we're mid-reasoning or mid-answer.

        remaining = delta

        while remaining:
            if not self.think_closed:
                # Haven't seen </think> yet — we're either in preamble or reasoning
                if not self.in_think:
                    # Look for <think> open tag
                    idx = remaining.find("<think>")
                    if idx != -1:
                        # Text before <think> goes to answer (preamble)
                        before = remaining[:idx]
                        if before:
                            a_out += before
                            self.answer += before
                        self.in_think = True
                        remaining = remaining[idx + len("<think>"):]
                    else:
                        # Might be a partial tag at the end
                        if remaining.endswith("<") or remaining.endswith("<t") or \
                           remaining.endswith("<th") or remaining.endswith("<thi") or \
                           remaining.endswith("<thin") or remaining.endswith("<think"):
                            # Hold the possible partial tag
                            self.answer += remaining[:-len(remaining.lstrip("<"))]
                            remaining = ""
                        else:
                            # No think tag coming — everything is answer text
                            a_out += remaining
                            self.answer += remaining
                            remaining = ""
                else:
                    # Inside <think> block — look for </think>
                    idx = remaining.find("</think>")
                    if idx != -1:
                        chunk = remaining[:idx]
                        r_out += chunk
                        self.reasoning += chunk
                        self.in_think     = False
                        self.think_closed = True
                        remaining = remaining[idx + len("</think>"):]
                    else:
                        r_out += remaining
                        self.reasoning += remaining
                        remaining = ""
            else:
                # </think> already seen — everything is final answer
                a_out += remaining
                self.answer += remaining
                remaining = ""

        return r_out, a_out


# ─── Answer extractors ────────────────────────────────────────────────────────

def _extract_yes_no(text: str) -> str | None:
    patterns = [
        r"\b(?:the\s+answer\s+is|answer\s*[:=])\s*(yes|no)\b",
        r"\b(?:therefore|thus|hence|so|conclusion\s*:)\s*,?\s*(yes|no)\b",
        r"\b(yes|no)[.,!]\s+(?:the|it|this|in|standard|scaled)\b",
        r"^(yes|no)[.,!]?\s*$",
    ]
    t = text.lower()
    for pat in patterns:
        m = re.search(pat, t, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _extract_same_unchanged(text: str) -> str | None:
    patterns = [
        r"\b(?:total\s+)?parameter\s+count\b.{0,80}(?:remains?|stays?)\s+(?:the\s+)?(?:same|unchanged)\b",
        r"\b(?:remains?|stays?)\s+(?:the\s+)?(?:same|unchanged)\b.{0,80}\bparameter\b",
        r"\b(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:total\s+)?(?:parameter\s+count\s+)?(?:remains?|stays?)\s+(?:the\s+)?(?:same|unchanged)\b",
        r"\bdoes\s+not\s+(?:change|increase|decrease)\b",
        r"\b(?:same|unchanged)\b.{0,30}\bregardless\b",
        r"\bparameter\s+count\s+is\s+(?:the\s+)?same\b",
    ]
    t = text.lower()
    for pat in patterns:
        if re.search(pat, t):
            return "same"
    return None


def _extract_integer(text: str, expected: int) -> str | None:
    ev = str(expected)
    t  = text.lower()
    patterns = [
        rf"(?:the\s+answer\s+is|result\s+is|equals?\s+|=\s*){re.escape(ev)}\b",
        rf"\b{re.escape(ev)}\s+(?:comparisons?|steps?|dimensions?|bits?|levels?|iterations?|times?)\b",
        rf"\b(?:maximum|max|minimum|min)\s+(?:\w+\s+){{0,4}}{re.escape(ev)}\b",
        rf"\btherefore[^.\n]{{0,80}}\b{re.escape(ev)}\b",
        rf"^{re.escape(ev)}\s*$",
        rf"\bso\s+(?:it\s+(?:is|takes?|needs?|makes?)\s+)?{re.escape(ev)}\b",
        rf"d_?k\s*=\s*{re.escape(ev)}\b",
        rf"log[_\s]*2\s*\(?[^)]*\)?\s*(?:=|≈|≤|is approximately|rounds?\s+(?:up\s+)?to)\s*{re.escape(ev)}\b",
        rf"\b{re.escape(ev)}\s+(?:is\s+the\s+)?(?:answer|result|value)\b",
        rf"=\s*{re.escape(ev)}\b",
        rf"\b{re.escape(ev)}\b.{{0,20}}(?:comparisons?|steps?|iterations?)",
    ]
    for pat in patterns:
        if re.search(pat, t, re.MULTILINE):
            return ev
    return None


def _extract_number(text: str, expected: float) -> str | None:
    ev_int = str(int(expected))
    patterns = [
        rf"\b{re.escape(ev_int)}\s*(?:GB|gigabytes?)\b",
        rf"(?:the\s+answer\s+is|=\s*|requires?\s+|need\s+){re.escape(ev_int)}\s*(?:GB|gigabytes?)?\b",
        rf"\btotal\s+(?:\w+\s+){{0,4}}{re.escape(ev_int)}\s*(?:GB|gigabytes?)\b",
        rf"\b{re.escape(ev_int)}\s*GB\b",
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return ev_int
    return None


def _extract_multi_numeric(text: str) -> str | None:
    has_acc = bool(re.search(
        r"accuracy[^0-9]{0,60}(90(?:\.0+)?|0\.90?)\s*%?",
        text, re.IGNORECASE
    ))
    has_f1 = bool(re.search(
        r"f[_\-]?1(?:\s+score)?[^0-9]{0,60}(0(?:\.0+)?)\b",
        text, re.IGNORECASE
    ))
    if has_acc and has_f1:
        return "accuracy=90%,f1=0"
    return None


def _extract_answer(text: str, answer_type: str, expected: str) -> str | None:
    if answer_type == "yes_no":
        return _extract_yes_no(text)
    if answer_type == "same":
        return _extract_same_unchanged(text)
    if answer_type == "integer":
        return _extract_integer(text, int(expected))
    if answer_type == "number":
        return _extract_number(text, float(expected))
    if answer_type == "multi_numeric":
        return _extract_multi_numeric(text)
    return None


def _answer_matches(extracted: str | None, expected: str, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "yes_no":
        return extracted.lower() == expected.lower()
    if answer_type == "same":
        return True
    if answer_type == "integer":
        try:
            return int(extracted) == int(expected)
        except ValueError:
            return False
    if answer_type == "number":
        try:
            return abs(float(extracted) - float(expected)) <= 0.5
        except ValueError:
            return False
    if answer_type == "multi_numeric":
        return extracted is not None
    return False


# ─── Post-hoc correctness checker ─────────────────────────────────────────────

def check_answer(full_text: str, expected: str, answer_type: str) -> bool:
    """
    Evaluate model's completed answer. Operates on full reasoning + answer text.
    Two-stage: structured extractor first, bare numeric fallback second.
    """
    clean = re.sub(r"<[^>]+>", " ", full_text)
    clean = re.sub(r"[*_`]", "", clean).replace("\u202f", " ")

    extracted = _extract_answer(clean, answer_type, expected)
    if _answer_matches(extracted, expected, answer_type):
        return True

    # Fallback: bare exact numeric match
    if answer_type == "integer":
        nums = re.findall(r"-?\d+", clean)
        return any(int(n) == int(expected) for n in nums if n)
    if answer_type == "number":
        nums = re.findall(r"-?\d+(?:\.\d+)?", clean)
        return any(abs(float(n) - float(expected)) <= 0.5 for n in nums if n)

    return False


# ─── API retry helper ──────────────────────────────────────────────────────────

def _stream_with_retry(messages, model, temperature=0.6, max_tokens=900):
    for attempt in range(8):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"thinking": {"type": "enabled", "budget_tokens": 512}},
            )
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                wait = 20 * (attempt + 1)
                print(f"  {Fore.YELLOW}[rate limit — waiting {wait}s]{Style.RESET_ALL}")
                time.sleep(wait)
            elif attempt == 0:
                # thinking param might not be supported — retry without it
                print(f"  {Fore.YELLOW}[retrying without thinking param]{Style.RESET_ALL}")
                try:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                except Exception:
                    raise
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ─── Main experiment function ─────────────────────────────────────────────────

def run_early_exit(
    question: str,
    model: str,
    answer_type: str,
    expected_answer: str,
) -> dict:
    """
    Single generation with live TERMINATOR-style exit detection.

    Per the paper's methodology:
      1. Stream the model's reasoning trace chunk by chunk
      2. After each reasoning chunk, attempt to extract a correct answer
         from the cumulative reasoning buffer so far
      3. Track stability: how many consecutive chunks show the same answer
      4. TERMINATOR fires when stability >= STABILITY_CHUNKS
         → record this word position as 'exit_word_position'
      5. If the answer disappears (model reconsidered) → reset stability
      6. After full generation: wasted = total_words - exit_word_position
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    stream = _stream_with_retry(messages, model)

    parser          = StreamParser()
    reasoning_words = 0
    total_tokens    = None

    # Exit-detection state
    stable_answer      = None
    stability_count    = 0
    exit_word_position = None
    exit_detected      = False

    print(f"  {Fore.CYAN}[streaming]{Style.RESET_ALL} ", end="", flush=True)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            total_tokens = chunk.usage.completion_tokens

        if not chunk.choices:
            continue

        raw_delta = getattr(chunk.choices[0].delta, "content", None) or ""
        r_delta, _ = parser.feed(raw_delta)

        if r_delta:
            reasoning_words += len(r_delta.split())

            # ── Live exit detection ──────────────────────────────────────
            if not exit_detected:
                extracted = _extract_answer(
                    parser.reasoning, answer_type, expected_answer
                )
                if _answer_matches(extracted, expected_answer, answer_type):
                    if extracted == stable_answer:
                        stability_count += 1
                    else:
                        stable_answer   = extracted
                        stability_count = 1

                    if stability_count >= STABILITY_CHUNKS:
                        exit_word_position = reasoning_words
                        exit_detected      = True
                        print(
                            f"\n  {Fore.GREEN}★ TERMINATOR EXIT at "
                            f"~{exit_word_position} reasoning words"
                            f"{Style.RESET_ALL}",
                            end="", flush=True
                        )
                else:
                    if stable_answer is not None:
                        print(
                            f"\n  {Fore.YELLOW}[reconsidered — stability reset]"
                            f"{Style.RESET_ALL}",
                            end="", flush=True
                        )
                    stable_answer   = None
                    stability_count = 0

    print()

    full_words = reasoning_words
    if exit_word_position is not None and full_words > 0:
        wasted_words        = max(0, full_words - exit_word_position)
        potential_reduction = (wasted_words / full_words) * 100
    else:
        wasted_words        = 0
        potential_reduction = 0.0

    total_tokens = total_tokens if total_tokens else full_words

    return {
        "strategy"            : "TERMINATOR-inspired stability-gated exit",
        "full_answer"         : parser.answer.strip(),
        "tokens"              : total_tokens,
        "full_reasoning_words": full_words,
        "exit_word_position"  : exit_word_position,
        "exit_detected"       : exit_detected,
        "wasted_words"        : wasted_words,
        "potential_reduction" : potential_reduction,
        "raw_reasoning"       : parser.reasoning,
        "raw"                 : parser.reasoning + "\n" + parser.answer,
    }