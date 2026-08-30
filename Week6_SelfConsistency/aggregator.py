# aggregator.py
from collections import Counter
import re

def normalize(answer: str) -> str:
    """
    Normalize answers for comparison — aggressively extract the numeric
    or short-form core so that "The ball costs $0.05", "$0.05", and "0.05"
    all collapse to the same bucket.
    """
    a = answer.lower().strip()
    a = re.sub(r'\*+', '', a)            # strip markdown bold
    a = re.sub(r'[,](?=\d{3})', '', a)   # strip thousands commas: 1,024 → 1024
    a = re.sub(r'\\?[$€£]', '', a)       # strip currency symbols
    a = re.sub(r'[^a-z0-9\s.\-/]', '', a)  # strip all other punctuation

    # Try to extract a number (integer or decimal) — this is the key fix.
    # For math-heavy self-consistency, the answer is almost always numeric.
    nums = re.findall(r'-?\d+\.?\d*', a)
    if nums:
        # Return the *last* number — models often restate the question's
        # numbers before giving the answer at the end.
        return nums[-1]

    # Non-numeric fallback: strip filler words, keep core
    a = re.sub(r'\b(the|answer|is|equals?|approximately|about|costs?|ball)\b', '', a)
    a = re.sub(r'\s+', ' ', a).strip()
    return a[:50]


def majority_vote(paths: list[dict]) -> dict:
    """
    Select the most consistent answer by majority vote.
    This is the core of self-consistency — marginalize over reasoning paths.
    """
    normalized = [(p['answer'], normalize(p['answer'])) for p in paths]
    counts = Counter(norm for _, norm in normalized)
    most_common_norm, vote_count = counts.most_common(1)[0]

    # Find the original answer that matches the most common normalized form
    winning_answer = next(
        orig for orig, norm in normalized if norm == most_common_norm
    )

    # Build vote distribution
    vote_dist = {}
    for orig, norm in normalized:
        vote_dist[norm] = vote_dist.get(norm, {"count": 0, "example": orig})
        vote_dist[norm]["count"] += 1

    return {
        "winner": winning_answer,
        "winner_normalized": most_common_norm,
        "votes": vote_count,
        "total": len(paths),
        "confidence": vote_count / len(paths),
        "distribution": dict(sorted(
            vote_dist.items(), key=lambda x: x[1]["count"], reverse=True
        ))
    }


def greedy_answer(paths: list[dict]) -> str:
    """Simulate greedy decoding — just take the first path's answer."""
    return paths[0]['answer'] if paths else ""