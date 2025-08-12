# TODO: implement save/load and search; wrap FAISS with metadata (source_url, title, section)
# src/index/faiss_store.py
from pathlib import Path
from typing import List, Dict, Tuple
import faiss, json, numpy as np

class FaissStore:
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index = None
        self.meta: List[Dict] = []

    def build(self, embeddings: np.ndarray, meta: List[Dict]):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine since we normalized
        self.index.add(embeddings)
        self.meta = meta

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.meta = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(self, query_emb: np.ndarray, top_k=5) -> List[Tuple[float, Dict]]:
        D, I = self.index.search(query_emb.astype("float32"), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            results.append((float(score), self.meta[idx]))
        return results
