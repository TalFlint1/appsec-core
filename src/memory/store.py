# src/memory/store.py
from pathlib import Path
from typing import List, Dict, Optional
import json, time
import numpy as np

class MemoryStore:
    def __init__(self, path="data/memory/memories.jsonl", embedder=None):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def add(self, text: str, tags=None):
        rec = {"ts": time.time(), "text": text, "tags": tags or [], "importance": 3, "ttl_days": 365, "category": "preference"}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._items.append(rec); self._embs = None

    # NEW: add with metadata (from manager)
    def add_with_meta(self, rec: Dict):
        if "ts" not in rec: rec["ts"] = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._items.append(rec); self._embs = None

    def all(self) -> List[Dict]:
        return list(self._items)

    def clear(self):
        self.path.write_text("")
        self._items.clear(); self._embs = None

    # Optional: basic semantic search if you want direct store.search access
    def _ensure_embs(self):
        if self._embs is not None: return
        texts = [it["text"] for it in self._items]
        if not texts:
            self._embs = np.zeros((0, 384), dtype="float32"); return
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        self._embs = np.asarray(embs, dtype="float32")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        self._ensure_embs()
        if self._embs.shape[0] == 0: return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = (self._embs @ q).tolist()
        idxs = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k]
        return [self._items[i]["text"] for i in idxs]
