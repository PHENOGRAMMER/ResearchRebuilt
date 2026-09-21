# rag_pipeline.py
# Week 9 — #ResearchRebuilt
# Retrieval-Augmented Generation (Lewis et al., NeurIPS 2020)
#
# Three strategies, side-by-side — mirrors strategies.py from Week 7:
#
#   Strategy 1 — Closed Book   : LLM answers from parametric memory only (no retrieval)
#   Strategy 2 — Sparse RAG    : BM25 keyword retrieval + LLM generation
#   Strategy 3 — Dense RAG     : DPR-style dense retrieval + LLM generation (paper's method)
#
# The paper's key claim: dense retrieval + generation > sparse retrieval > closed-book
# on knowledge-intensive tasks, especially for facts that change or aren't memorised.

from groq import Groq
from retriever import retrieve_dense, retrieve_bm25_approx
import os, re, time
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

# ── Shared utilities ──────────────────────────────────────────────────────────

# FIX 1 — Force structured output so _extract_answer always finds the answer
# The model was generating markdown tables / bullet lists, hitting 400 tokens,
# and getting truncated mid-sentence. Two fixes:
#   (a) Explicit FINAL ANSWER: <value> instruction in every prompt
#   (b) max_tokens raised to 600 so calculation questions don't get cut off

ANSWER_INSTRUCTION = (
    "\n\nIMPORTANT: End your response with exactly this format on its own line:\n"
    "FINAL ANSWER: <your concise answer here>"
)


def _call(messages: list[dict], temperature: float = 0.0) -> tuple[str, int]:
    """Shared API call with rate-limit retry — matches Week 7 pattern."""
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=600,          # FIX 1b — was 400, now 600
            )
            return resp.choices[0].message.content, resp.usage.completion_tokens
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 12 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq API: max retries exceeded")


def _extract_answer(text: str) -> str:
    """
    Pull final answer from model output.
    Priority: FINAL ANSWER label → any answer label → last non-empty line.
    FIX 2 — now prefers 'FINAL ANSWER:' which we inject into every prompt.
    """
    # Priority 1: explicit FINAL ANSWER label (our injected format)
    m = re.search(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()[:300]

    # Priority 2: any answer/conclusion label
    labeled = re.findall(
        r"(?:final answer|answer|therefore|thus|conclusion)[:\-\s]+(.+)",
        text, flags=re.IGNORECASE
    )
    if labeled:
        return labeled[-1].strip()[:300]

    # Fallback: last non-empty, non-separator line
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not re.fullmatch(r"[-=|*\s]+", l.strip())]
    return lines[-1][:300] if lines else text[:300]


# ── Strategy 1: Closed Book (no retrieval) ───────────────────────────────────

def closed_book(question: str) -> dict:
    """
    Baseline: pure parametric memory. No retrieval, no external context.
    This is what all LLMs do by default.
    Paper finding: fails on knowledge-intensive tasks — hallucinates, gives outdated facts.
    """
    prompt = (
        f"Answer the following question as accurately as possible.\n\n"
        f"Question: {question}"
        f"{ANSWER_INSTRUCTION}"
    )
    text, tokens = _call([{"role": "user", "content": prompt}])
    return {
        "strategy":  "Closed Book (parametric only)",
        "answer":    _extract_answer(text),
        "tokens":    tokens,
        "docs_used": [],
        "raw":       text,
    }


# ── Strategy 2: Sparse RAG (BM25 + generation) ───────────────────────────────

def sparse_rag(question: str, k: int = 3) -> dict:
    """
    Sparse retrieval baseline: approximate BM25 keyword matching, then generation.
    The paper compared against BM25 retrievers and showed dense retrieval is significantly
    better on knowledge-intensive NLP tasks (Table 1 in the paper).
    """
    docs = retrieve_bm25_approx(question, k=k)
    context = "\n\n".join(
        f"[Document {i+1}] {d['text']}" for i, d in enumerate(docs)
    )
    prompt = (
        f"Use ONLY the following retrieved documents to answer the question.\n"
        f"Do not use external knowledge — only what is in the documents below.\n\n"
        f"Retrieved Documents:\n{context}\n\n"
        f"Question: {question}"
        f"{ANSWER_INSTRUCTION}"
    )
    text, tokens = _call([{"role": "user", "content": prompt}])
    return {
        "strategy":  "Sparse RAG (BM25 + generation)",
        "answer":    _extract_answer(text),
        "tokens":    tokens,
        "docs_used": [d["text"][:80] + "…" for d in docs],
        "scores":    [round(d["score"], 3) for d in docs],
        "raw":       text,
    }


# ── Strategy 3: Dense RAG (DPR-style + generation) ───────────────────────────

def dense_rag(question: str, k: int = 3) -> dict:
    """
    Paper's method: dense retrieval (DPR bi-encoder MIPS) + conditional generation.

    RAG-Sequence formulation (paper Eq. 1):
        p_RAG(y|x) ≈ Σ_{z∈top-k} p_η(z|x) · p_θ(y|x,z)

    We implement the RAG-Sequence variant:
    same top-k documents condition the entire generated answer.
    The retrieval score p_η(z|x) = exp(d(z)ᵀq(x)) is the dense similarity.
    """
    docs = retrieve_dense(question, k=k)
    context = "\n\n".join(
        f"[Document {i+1} | relevance: {d['score']:.3f}]\n{d['text']}"
        for i, d in enumerate(docs)
    )
    prompt = (
        f"You have access to the following retrieved documents, ranked by relevance.\n"
        f"Use them to answer the question. "
        f"Prefer information from the documents over your internal knowledge.\n\n"
        f"Retrieved Documents:\n{context}\n\n"
        f"Question: {question}"
        f"{ANSWER_INSTRUCTION}"
    )
    text, tokens = _call([{"role": "user", "content": prompt}])
    return {
        "strategy":  "Dense RAG (DPR-style + generation)",
        "answer":    _extract_answer(text),
        "tokens":    tokens,
        "docs_used": [d["text"][:80] + "…" for d in docs],
        "scores":    [round(d["score"], 3) for d in docs],
        "raw":       text,
    }