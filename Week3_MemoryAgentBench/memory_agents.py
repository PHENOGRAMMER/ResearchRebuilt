# memory_agents.py
from rank_bm25 import BM25Okapi
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class BM25MemoryAgent:
    """Simple keyword-based memory — fast but no semantic understanding."""
    def __init__(self):
        self.memories = []      # list of text strings
        self.tokenized = []

    def add(self, text: str):
        self.memories.append(text)
        self.tokenized.append(text.lower().split())

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.memories:
            return []
        bm25 = BM25Okapi(self.tokenized)
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.memories[i] for i in top_idx if scores[i] > 0]

    def forget(self, query: str):
        """Selective forgetting — remove memories matching query keywords."""
        keywords = set(query.lower().split())
        self.memories = [
            m for m in self.memories
            if not keywords.intersection(set(m.lower().split()))
        ]
        self.tokenized = [m.lower().split() for m in self.memories]

    def update(self, old_fact_keyword: str, new_fact: str):
        """Update a fact by removing old and adding new."""
        self.forget(old_fact_keyword)
        self.add(new_fact)

    def clear(self):
        self.memories = []
        self.tokenized = []


class SemanticMemoryAgent:
    """FAISS-backed semantic memory — understands meaning, not just keywords."""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memories = []
        self.index = None

    def _rebuild(self):
        if not self.memories:
            self.index = None
            return
        embs = self.model.encode(self.memories).astype('float32')
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)

    def add(self, text: str):
        self.memories.append(text)
        self._rebuild()

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.memories or self.index is None:
            return []
        q = self.model.encode([query]).astype('float32')
        k = min(top_k, len(self.memories))
        D, I = self.index.search(q, k)
        return [self.memories[i] for i in I[0] if i < len(self.memories)]

    def forget(self, fact_to_remove: str):
        """Remove semantically similar memories."""
        if not self.memories:
            return
        q = self.model.encode([fact_to_remove]).astype('float32')
        if self.index:
            D, I = self.index.search(q, min(3, len(self.memories)))
            to_remove = set(I[0])
            self.memories = [m for i, m in enumerate(self.memories) if i not in to_remove]
            self._rebuild()

    def update(self, old_fact_keyword: str, new_fact: str):
        self.forget(old_fact_keyword)
        self.add(new_fact)

    def clear(self):
        self.memories = []
        self.index = None


class HybridMemoryAgent:
    """Combines BM25 and Semantic memory using Reciprocal Rank Fusion (RRF)."""
    def __init__(self):
        self.bm25 = BM25MemoryAgent()
        self.semantic = SemanticMemoryAgent()

    def add(self, text: str):
        self.bm25.add(text)
        self.semantic.add(text)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        bm25_res = self.bm25.retrieve(query, top_k=top_k * 2)
        sem_res = self.semantic.retrieve(query, top_k=top_k * 2)
        
        scores = {}
        for rank, doc in enumerate(bm25_res):
            scores[doc] = scores.get(doc, 0) + 1.0 / (60 + rank)
        for rank, doc in enumerate(sem_res):
            scores[doc] = scores.get(doc, 0) + 1.0 / (60 + rank)
            
        sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_docs[:top_k]

    def forget(self, fact_to_remove: str):
        self.bm25.forget(fact_to_remove)
        self.semantic.forget(fact_to_remove)

    def update(self, old_fact_keyword: str, new_fact: str):
        self.bm25.update(old_fact_keyword, new_fact)
        self.semantic.update(old_fact_keyword, new_fact)

    def clear(self):
        self.bm25.clear()
        self.semantic.clear()


class MultiHopMemoryAgent(SemanticMemoryAgent):
    """Uses LLM query expansion to rewrite queries for multi-hop retrieval."""
    def __init__(self):
        super().__init__()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def _expand_query(self, query: str) -> list[str]:
        prompt = f"""You are a search query expansion expert.
The user is asking: '{query}'
Rewrite this query into 2-3 simpler, declarative search queries that would help find the information piece by piece.
For example, if the query is "What university did the CEO attend?", you might output:
Who is the CEO?
What university did they attend?
Only output the queries, one per line. Do not output anything else."""
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            queries = [q.strip() for q in resp.choices[0].message.content.strip().split('\\n') if q.strip()]
            return queries if queries else [query]
        except Exception as e:
            return [query]

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.memories or self.index is None:
            return []
            
        queries = [query] + self._expand_query(query)
        results = []
        for q in queries:
            results.extend(super().retrieve(q, top_k=top_k))
            
        # Deduplicate while preserving order
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
                
        return unique_results[:top_k * 2] # return slightly larger context