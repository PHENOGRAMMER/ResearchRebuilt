# memory_bank.py
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

class MemoryBank:
    def __init__(self):
        self.entries = []   # list of strings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None

    def _rebuild(self):
        if not self.entries:
            self.index = None
            return
        embs = self.model.encode(self.entries).astype('float32')
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)

    def add(self, text: str):
        self.entries.append(text)
        self._rebuild()

    def update(self, old_idx: int, new_text: str):
        if 0 <= old_idx < len(self.entries):
            self.entries[old_idx] = new_text
            self._rebuild()

    def delete(self, old_idx: int):
        if 0 <= old_idx < len(self.entries):
            self.entries.pop(old_idx)
            self._rebuild()

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, str]]:
        if not self.entries or self.index is None:
            return []
        q = self.model.encode([query]).astype('float32')
        k = min(top_k, len(self.entries))
        D, I = self.index.search(q, k)
        return [(i, self.entries[i]) for i in I[0] if i < len(self.entries)]

    def show(self):
        print(f"  Memory Bank ({len(self.entries)} entries):")
        for i, e in enumerate(self.entries):
            print(f"    [{i}] {e}")