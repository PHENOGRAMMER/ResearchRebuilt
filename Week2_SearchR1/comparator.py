"""
comparator.py

Two baselines compared against Search-R1 (mirrors Table 1 in arXiv:2503.09516):

  No Retrieval  — LLM answers purely from parametric memory (weights).
                  Corresponds to the "Vanilla LLM" row in the paper.

  Naive RAG     — Single retrieval pass, no reasoning loop.
                  Corresponds to "Standard RAG" in the paper.
                  The retrieval query is a simple keyword extraction from the
                  question — no iterative refinement, no chain-of-thought.

Both use the same underlying model as the Search-R1 agent so that the only
variable is the retrieval strategy.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_MODEL = "llama-3.1-8b-instant"


def run_no_retrieval(query: str) -> str:
    """
    Baseline 1 — Vanilla LLM (no retrieval).
    The model must answer entirely from weights. For multi-hop factual
    questions this consistently fails or hallucinates, which is the
    point of the comparison.
    """
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable assistant. Answer questions concisely and factually.",
            },
            {
                "role": "user",
                "content": f"Answer this question as accurately as possible: {query}",
            },
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def run_naive_rag(query: str, retrieved_context: str) -> str:
    """
    Baseline 2 — Standard RAG (single retrieval, single generation).
    The model sees one retrieved passage and must answer from it directly.
    No reasoning loop, no follow-up searches.

    This is the closest approximation to a retrieval-augmented generation
    baseline without any chain-of-thought or iterative refinement.
    """
    # Truncate context to avoid blowing the context window
    context_trimmed = retrieved_context[:2000]

    prompt = (
        "Use the provided context to answer the question. "
        "If the context does not contain enough information to answer, "
        "say so clearly rather than guessing.\n\n"
        f"Context:\n{context_trimmed}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise question-answering assistant. Answer using only the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()