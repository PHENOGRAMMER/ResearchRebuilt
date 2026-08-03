# retriever.py
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HERA-Recreation/1.0'
)

def search_wikipedia(query, max_sentences=5):
    """Fetch top Wikipedia content for a query."""
    page = wiki.page(query)
    if not page.exists():
        # Try simpler keyword
        keyword = query.split()[0]
        page = wiki.page(keyword)
    if page.exists():
        sentences = page.summary.split('. ')[:max_sentences]
        return '. '.join(sentences)
    return "No Wikipedia article found."