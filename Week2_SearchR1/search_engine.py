import wikipediaapi
import urllib.parse
import urllib.request
import json
import re

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="SearchR1-Recreation/1.0 (research project)"
)

_SEARCH_CACHE = {}

# Words that carry no retrieval signal
_STOP_WORDS = {
    "what", "where", "when", "who", "which", "whose", "with", "this", "that",
    "about", "find", "search", "query", "name", "named", "also", "from",
    "have", "does", "did", "the", "and", "for", "are", "was", "were", "how",
    "its", "been", "being", "into", "than", "then", "they", "them", "their",
    "year", "most", "more", "some", "such", "only", "each", "both", "very",
}

# Section headings that commonly hold the exact fact we need
_BOOST_SECTIONS = {
    "education", "early life", "career", "history", "founding", "founded",
    "university", "degree", "geography", "height", "elevation", "altitude",
    "unesco", "heritage", "capital", "overview", "background", "origin",
    "creation", "development", "publications", "awards", "recognition",
    "demographics", "economy", "government", "politics", "description",
    "establishment", "founding year", "location", "features", "summary",
}


# ---------------------------------------------------------------------------
# Wikipedia Search API — returns ranked title list
# ---------------------------------------------------------------------------

def _wikipedia_search_api(query: str, limit: int = 10) -> list[str]:
    """Hit the Wikipedia search API and return up to `limit` article titles."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={encoded}"
        f"&format=json&srlimit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SearchR1-Recreation/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            return [hit["title"] for hit in data["query"]["search"]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, remove stop words, keep tokens > 2 chars."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


def _term_overlap(tokens_a: list[str], tokens_b: list[str]) -> int:
    """Number of shared tokens between two token lists."""
    set_b = set(tokens_b)
    return sum(1 for t in tokens_a if t in set_b)


# ---------------------------------------------------------------------------
# Section walker — recursively collects (score, title, text) tuples
# ---------------------------------------------------------------------------

def _collect_sections(
    sections,
    query_tokens: list[str],
) -> list[tuple[int, str, str]]:
    results = []
    for sec in sections:
        title_lower = sec.title.lower()
        score = 0

        # Boost for well-known informative section names
        if any(kw in title_lower for kw in _BOOST_SECTIONS):
            score += 3

        # Additional boost if query terms appear in the section title
        sec_title_tokens = _tokenize(sec.title)
        score += _term_overlap(query_tokens, sec_title_tokens) * 2

        # Light boost if query terms appear in the section body
        body_tokens = _tokenize(sec.text)
        body_overlap = _term_overlap(query_tokens, body_tokens)
        score += min(body_overlap, 5)  # cap to avoid drowning title signal

        if score > 0 and len(sec.text.strip()) > 60:
            results.append((score, sec.title, sec.text))

        if sec.sections:
            results.extend(_collect_sections(sec.sections, query_tokens))

    return results


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search(query: str, max_sentences: int = 8, max_sections: int = 2) -> str:

    normalized_query = query.lower().strip()
    if normalized_query in _SEARCH_CACHE:
        return _SEARCH_CACHE[normalized_query]

    """
    Search Wikipedia for `query` and return a formatted snippet.

    Strategy:
    1. Call the Wikipedia search API to get candidate titles.
    2. Score each candidate article based on title overlap, exact-phrase match,
       and summary term density.
    3. Return the best article's summary + top relevant sections.
    4. Fall back to a direct Wikipedia page lookup if scoring fails.
    """
    titles = _wikipedia_search_api(query, limit=12)
    query_tokens = _tokenize(query)
    entities = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", query)

    query_lower = query.lower()

    scored_pages: list[tuple[int, object, str]] = []

    for title in titles:
        # Skip disambiguation pages — they never have the answer directly
        if "(disambiguation)" in title.lower():
            continue

        page = wiki.page(title)
        if (
            not page.exists()
            or len(page.summary) < 120
            or len(page.sections) == 0
        ):
            continue

        score = 0
        title_lower = title.lower()
        title_tokens = _tokenize(title)

        # ── Title token overlap ───────────────────────────────────────────────
        overlap = _term_overlap(title_tokens, query_tokens)
        # -----------------------------
        # Title match
        # -----------------------------
        overlap = _term_overlap(title_tokens, query_tokens)
        score += overlap * 20

        # -----------------------------
        # Exact / partial title match
        # -----------------------------
        if title_lower == query_lower:
            score += 120

        elif title_lower in query_lower:
            score += 70

        elif query_lower in title_lower:
            score += 40

        # -----------------------------
        # Consecutive query words
        # -----------------------------
        for i in range(len(query_tokens) - 1):
            bigram = query_tokens[i] + " " + query_tokens[i + 1]

            if bigram in title_lower:
                score += 20

        # -----------------------------
        # Entity overlap
        # -----------------------------
        entity_overlap = 0

        for entity in entities:
            if entity.lower() in page.summary.lower():
                entity_overlap += 1

        score += entity_overlap * 25

        # -----------------------------
        # Summary relevance
        # -----------------------------
        summary_tokens = _tokenize(page.summary)
        summary_overlap = _term_overlap(query_tokens, summary_tokens)

        score += min(summary_overlap * 3, 30)

        scored_pages.append((score, page, title))

    # -------------------------------------------------------
    # Pick highest scoring page
    # -------------------------------------------------------
    if scored_pages:

        scored_pages.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        best_score, page, title = scored_pages[0]

    else:

        if titles:

            page = wiki.page(titles[0])
            title = titles[0]

            if not page.exists():

                result = (
                    f"[Search Failure]\n"
                    f"No reliable Wikipedia page found for '{query}'."
                )

                _SEARCH_CACHE[normalized_query] = result

                return result

        else:

            page = wiki.page(query)
            title = query

            if not page.exists():

                result = (
                    f"[Search Failure]\n"
                    f"No reliable Wikipedia page found for '{query}'."
                )

                _SEARCH_CACHE[normalized_query] = result

                return result

    # -------------------------------------------------------
    # Build retrieval context
    # -------------------------------------------------------

    content_parts = []

    # Summary
    sentences = [
        s.strip()
        for s in page.summary.split(".")
        if s.strip()
    ]

    summary_text = ". ".join(sentences[:max_sentences]) + "."

    content_parts.append(summary_text)

    # Relevant sections
    relevant_sections = _collect_sections(
        page.sections,
        query_tokens,
    )

    relevant_sections.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for _, sec_title, sec_text in relevant_sections[:3]:

        paragraphs = [
            p.strip()
            for p in sec_text.split("\n")
            if p.strip()
        ]

        excerpt = "\n".join(paragraphs[:3])

        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "..."

        content_parts.append(
            f"=== {sec_title} ===\n{excerpt}"
        )

    result = (
        f"[Source: {title}] "
        + "\n\n".join(content_parts)
    )

    _SEARCH_CACHE[normalized_query] = result

    return result