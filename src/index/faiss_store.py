# src/index/faiss_store.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Any
import json
import re
import numpy as np
import faiss
from utils.acronyms import expand_query_text, acronym_signal_boost

# ---- Optional BM25 import: runtime vs typing split ----
try:
    # Runtime import (may fail if rank-bm25 not installed)
    from rank_bm25 import BM25Okapi as BM25OkapiRuntime
except Exception:  # pragma: no cover
    BM25OkapiRuntime = None  # type: ignore

if TYPE_CHECKING:
    # Only for type checkers; does not run at runtime
    from rank_bm25 import BM25Okapi as BM25OkapiT
else:
    BM25OkapiT = Any  # Fallback type for annotations when package isn't present


def _simple_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class FaissStore:
    """
    Wraps a FAISS index for dense retrieval and (optionally) a BM25 index for
    keyword retrieval over the same corpus.

    Meta format expectation (per line in meta.jsonl):
        {
          "url": "...",
          "title": "...",
          "text": "full chunk text",
          ...  # any extra fields are preserved
        }
    """
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict] = []
        # --- BM25 (lazy) ---
        self._bm25: Optional[BM25OkapiT] = None
        self._bm25_docs: Optional[List[List[str]]] = None

    # -------- Dense index (FAISS) --------
    def build(self, embeddings: np.ndarray, meta: List[Dict]):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine if embeddings are L2-normalized
        self.index.add(embeddings.astype("float32"))
        self.meta = meta

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        D, I = self.index.search(query_emb.astype("float32"), top_k)
        results: List[Tuple[float, Dict]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[int(idx)]))
        return results
    
    # -------- Convenience: text → embed → FAISS (+ acronym handling) --------
    def _encode_query(self, embedder, query_texts: List[str]) -> np.ndarray:
        """
        Accepts either a SentenceTransformer-like object with .encode(...),
        or a callable that returns a np.ndarray for a list of texts.
        Must return shape (n, d) float32.
        """
        if hasattr(embedder, "encode"):
            vecs = embedder.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
        else:
            # embedder is a callable: def embed(list[str]) -> np.ndarray
            vecs = embedder(query_texts)
        vecs = np.asarray(vecs, dtype="float32")
        return vecs

    def search_text(self, query: str, embedder, top_k: int = 5, boost: bool = True) -> List[Tuple[float, Dict]]:
        """
        Text-first search with generic acronym support (CSRF/XSS/SSRF/SQLi/XXE/IDOR...).
        - Expands the query text so acronyms and full names are both represented.
        - Encodes the expanded query and searches FAISS.
        - Optionally applies a tiny content-based boost (URL-agnostic).
        Returns: List of (score, meta) like .search().
        """
        # 1) Expand acronymy queries (no URL assumptions)
        qx = expand_query_text(query)

        # 2) Encode expanded query and run FAISS
        qv = self._encode_query(embedder, [qx])
        results = self.search(qv, top_k=top_k)

        # 3) Optional: small content-based nudge (keeps original format)
        if boost and results:
            hits = [{"score": s, **m} for (s, m) in results]
            hits = acronym_signal_boost(query, hits, alpha=0.12)
            results = [(h["score"], {k: v for k, v in h.items() if k != "score"}) for h in hits]
        return results

    # -------- Keyword index (BM25) --------
    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        if BM25OkapiRuntime is None:
            raise RuntimeError("rank-bm25 is not installed. Please `pip install rank-bm25`.")
        texts: List[str] = []
        for m in self.meta:
            txt = m.get("text")
            if not txt:
                # fallback: combine title + sectionish fields if text missing
                pieces = [m.get("title", ""), m.get("section", ""), m.get("summary", "")]
                txt = " ".join([p for p in pieces if p])
            texts.append(txt or "")
        self._bm25_docs = [_simple_tokens(t) for t in texts]
        # type: ignore[arg-type] is safe because _bm25_docs is List[List[str]]
        self._bm25 = BM25OkapiRuntime(self._bm25_docs)  # type: ignore[call-arg]

    def search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[float, Dict]]:
        self._ensure_bm25()
        assert self._bm25 is not None and self._bm25_docs is not None
        q_tokens = _simple_tokens(query)
        scores = self._bm25.get_scores(q_tokens)  # numpy array length = n_docs
        # argsort descending and take top_k
        idxs = np.argsort(scores)[::-1][:top_k]
        out: List[Tuple[float, Dict]] = []
        for i in idxs:
            sc = float(scores[int(i)])
            if sc <= 0.0:
                # BM25 can return zeros for totally unrelated docs; skip tail
                continue
            out.append((sc, self.meta[int(i)]))
        return out

    # -------- Utilities --------
    @property
    def size(self) -> int:
        return len(self.meta)
