# src/memory/store.py
from pathlib import Path
from typing import List, Dict, Optional
import json, time
import numpy as np

class MemoryStore:
    """Tiny persistent memory: JSONL + semantic search via your embedder."""
    def __init__(self, path="data/memory/memories.jsonl", embedder=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._items: List[Dict] = []
        self._embs: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self._items.append(json.loads(line))
                    except Exception:
                        pass

    def add(self, text: str, tags: Optional[List[str]] = None):
        rec = {"ts": time.time(), "text": text, "tags": tags or []}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._items.append(rec)
        self._embs = None  # invalidate cache

    def all(self) -> List[Dict]:
        return list(self._items)

    def clear(self):
        self.path.write_text("")
        self._items.clear()
        self._embs = None

    def _ensure_embs(self):
        if self._embs is not None:
            return
        texts = [it["text"] for it in self._items]
        if not texts:
            self._embs = np.zeros((0, 384), dtype="float32")
            return
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        self._embs = np.asarray(embs, dtype="float32")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        self._ensure_embs()
        if self._embs.shape[0] == 0:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")[0]
        sims = self._embs @ q  # cosine similarity (embs are normalized)
        idxs = np.argsort(-sims)[:top_k]
        return [self._items[i]["text"] for i in idxs]
