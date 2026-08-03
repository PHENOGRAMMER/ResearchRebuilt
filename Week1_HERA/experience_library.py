# experience_library.py
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ExperienceLibrary:
    def __init__(self):
        self.entries = []  # list of {id, query_type, insight, utility}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # tiny, runs on CPU
        self.index = None
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.entries:
            self.index = None
            return
        embeddings = self.model.encode([e['insight'] for e in self.entries])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype='float32'))

    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant experiences for a query."""
        if not self.entries or self.index is None:
            return []
        q_emb = self.model.encode([query]).astype('float32')
        D, I = self.index.search(q_emb, min(top_k, len(self.entries)))
        return [self.entries[i] for i in I[0] if i < len(self.entries)]

    def add(self, query_type, insight):
        entry = {
            "id": len(self.entries),
            "query_type": query_type,
            "insight": insight,
            "utility": 0
        }
        self.entries.append(entry)
        self._rebuild_index()
        print(f"  [ExperienceLib] ADD: {insight[:80]}...")

    def update_utility(self, insight_text, success):
        for e in self.entries:
            if e['insight'] == insight_text and success:
                e['utility'] += 1

    def save(self, path="experience_library.json"):
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)
        print(f"  [ExperienceLib] Saved {len(self.entries)} entries.")

    def load(self, path="experience_library.json"):
        try:
            with open(path) as f:
                self.entries = json.load(f)
            self._rebuild_index()
            print(f"  [ExperienceLib] Loaded {len(self.entries)} entries.")
        except FileNotFoundError:
            print("  [ExperienceLib] Starting fresh.")