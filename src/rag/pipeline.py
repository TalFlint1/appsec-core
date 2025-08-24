# src/rag/pipeline.py
from __future__ import annotations
import re
import yaml
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

from sentence_transformers import SentenceTransformer, CrossEncoder

from .prompts import build_prompt, SYSTEM, GENERAL_SYSTEM
from ..index.faiss_store import FaissStore
from ..memory.store import MemoryStore
from ..memory.manager import MemoryManager
from ..utils.metrics import record_time

# ---------------- citation helpers (kept + extended) ----------------
def _unique_urls_in_order(chunks: List[Dict]) -> List[str]:
    seen, out = set(), []
    for ch in chunks:
        u = ch.get("url", "")
        if not u or not u.startswith("http"):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _diversify_chunks(hits: List[Tuple[float, Dict]], k: int, max_per_url: int) -> List[Dict]:
    """hits: list of (score, chunk) best->worst; keep at most max_per_url per URL."""
    out, counts = [], defaultdict(int)
    for _score, ch in hits:
        url = ch.get("url", "")
        if counts[url] >= max_per_url:
            continue
        out.append(ch)
        counts[url] += 1
        if len(out) >= k:
            break
    return out

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ACRONYM_MAP = {
    "ssrf": "server-side request forgery",
    "xss": "cross-site scripting",
    "csrf": "cross-site request forgery",
    "idor": "insecure direct object reference",
    "sqli": "sql injection",
    "ssti": "server-side template injection",
}

# Preferred domains (soft)
PREFERRED_CITATION_DOMAINS = (
    "owasp.org",
    "cheatsheetseries.owasp.org",
    "portswigger.net/web-security",
)

def _domain_rank(url: str) -> int:
    for i, d in enumerate(PREFERRED_CITATION_DOMAINS):
        if d in url:
            return i
    return len(PREFERRED_CITATION_DOMAINS)

def _topic_overlap(question: str, ch: dict) -> int:
    q = question.lower()
    title_url = (ch.get("title", "") + " " + ch.get("url", "")).lower()
    text = (ch.get("text", "") or "").lower()[:1200]

    terms = set(t for t in re.findall(r"[a-z0-9\-]+", q) if len(t) > 2)
    for k, v in ACRONYM_MAP.items():
        if k in terms:
            terms.add(v)
            terms.update(v.split())
            terms.add(v.replace(" ", "-"))

    score = 0
    for t in terms:
        if t in title_url:
            score += 2
        elif t in text:
            score += 1
    return score

def _select_citations(question: str, used_chunks: list, raw_hits: list, k: int) -> list[str]:
    sim_map = {}
    for s, ch in raw_hits:
        if isinstance(s, (int, float)):
            sim_map[id(ch)] = float(s)

    candidates = [ch for ch in used_chunks if ch.get("url", "").startswith("http")]
    scored = []
    for ch in candidates:
        url = ch["url"]
        tscore = _topic_overlap(question, ch)
        dscore = -_domain_rank(url)  # higher is better
        sscore = sim_map.get(id(ch), 0.0)
        scored.append(((tscore, dscore, sscore), url))

    scored.sort(reverse=True)  # topic > domain > similarity
    out, seen = [], set()
    for _, url in scored:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= k:
            break
    if not out:
        out = _unique_urls_in_order(candidates)[:k]
    return out
# -------------------------------------------------------------------

def _expand_acronyms(q: str) -> str:
    def repl(m):
        t = m.group(0)
        exp = ACRONYM_MAP.get(t.lower())
        return f"{t} ({exp})" if exp else t
    return re.sub(r"\b(SSRF|XSS|CSRF|IDOR|SQLi|SSTI)\b", repl, q, flags=re.IGNORECASE)

def _rrf_fuse(dense: List[Tuple[float, Dict]], bm25: List[Tuple[float, Dict]], k: int = 60) -> List[Tuple[float, Dict]]:
    """
    Reciprocal Rank Fusion over URL keys to avoid dup-chunk collisions.
    """
    def key_of(ch):
        return ch.get("url", "") + "::" + ch.get("title", "")
    ranks: Dict[str, float] = {}
    obj: Dict[str, Dict] = {}
    for lst in (dense, bm25):
        for rank, (_score, ch) in enumerate(lst, start=1):
            key = key_of(ch)
            obj[key] = ch
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (k + rank)
    fused = sorted([(score, obj[k]) for k, score in ranks.items()], key=lambda x: x[0], reverse=True)
    return fused

class RAGPipeline:
    def __init__(self, models_cfg="configs/models.yaml", index_dir="data/index"):
        self.cfg = load_yaml(models_cfg)
        self.index_dir = Path(index_dir)
        self.store = FaissStore(self.index_dir / "index.faiss", self.index_dir / "meta.jsonl")
        self.store.load()

        # Embeddings
        emb_model_name = self.cfg["embeddings"]["model_name"]
        emb_device = self.cfg["embeddings"].get("device", "cpu")
        self.embedder = SentenceTransformer(emb_model_name, device=emb_device)

        # Retrieval knobs
        r = self.cfg.get("retrieval", {}) or {}
        self.top_k                 = int(r.get("top_k", 5))
        self.candidate_multiplier  = int(r.get("candidate_multiplier", 5))
        self.max_chunks_per_url    = int(r.get("max_chunks_per_url", 2))
        self.citations_k           = int(r.get("citations_k", 1))  # default single cite
        self.min_hits_for_grounded = int(r.get("min_hits_for_grounded", 2))
        self.score_threshold       = float(r.get("score_threshold", 0.35))
        self.open_domain_fallback  = bool(r.get("open_domain_fallback", True))

        # Hybrid
        self.hybrid_enabled  = bool(r.get("hybrid_enabled", True))
        self.hybrid_fusion   = r.get("hybrid_fusion", "rrf")  # rrf | weighted
        self.bm25_topn       = int(r.get("bm25_topn", self.top_k * self.candidate_multiplier))
        self.bm25_weight     = float(r.get("bm25_weight", 0.35))  # only if weighted

        # Optional reranker
        self.use_rerank      = bool(r.get("use_rerank", False))
        self.reranker_model  = r.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker_top_n  = int(r.get("reranker_top_n", 30))
        self.reranker        = None  # lazy init

        # LLM
        self.llm_provider = self.cfg["llm"]["provider"]
        self.llm_model    = self.cfg["llm"]["model"]
        self.temperature  = float(self.cfg["llm"].get("temperature", 0.2))

        # Memory
        self.mem_store   = MemoryStore(embedder=self.embedder)
        self.mem         = self.mem_store
        self.mem_enabled = bool(r.get("mem_enabled", True))
        self.mem_k       = int(r.get("mem_k", 3))
        self.mem_manager = MemoryManager(
            store=self.mem_store,
            embedder=self.embedder,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            temperature=0.0,
        )

    # ----------------- helpers -----------------
    def _embed_query(self, q: str) -> np.ndarray:
        e = self.embedder.encode([q], normalize_embeddings=True)
        return np.asarray(e, dtype="float32")

    def _maybe_init_reranker(self):
        if not self.use_rerank:
            return
        if self.reranker is None:
            self.reranker = CrossEncoder(self.reranker_model, max_length=512)

    def _rerank(self, question: str, hits: List[Tuple[float, Dict]]):
        if not self.reranker:
            return hits
        top_n = min(self.reranker_top_n, len(hits))
        pairs = [(question, hits[i][1]["text"][:2000]) for i in range(top_n)]
        if not pairs:
            return hits
        scores = self.reranker.predict(pairs).tolist()
        reranked = sorted(zip(scores, [hits[i][1] for i in range(top_n)]),
                          key=lambda x: x[0], reverse=True)
        return [(s, ch) for s, ch in reranked]

    def _format_mem_block(self, snips):
        if not snips:
            return ""
        bullets = "\n".join(f"- {s}" for s in snips)
        return f"Known user facts (use only if helpful):\n{bullets}\n"

    def _call_llm(self, prompt: str, system: str = SYSTEM) -> str:
        if self.llm_provider == "ollama":
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": f"{system}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        elif self.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            chat = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return chat.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown llm provider: {self.llm_provider}")

    # ----------------- retrieval core -----------------
    def _dense_candidates(self, q_text: str, k: int) -> List[Tuple[float, Dict]]:
        q_emb = self._embed_query(q_text)
        return self.store.search(q_emb, top_k=k)

    def _bm25_candidates(self, q_text: str, k: int) -> List[Tuple[float, Dict]]:
        return self.store.search_bm25(q_text, top_k=k)

    def _hybrid_candidates(self, q_text: str, k: int) -> List[Tuple[float, Dict]]:
        dense = self._dense_candidates(q_text, k)
        try:
            bm25 = self._bm25_candidates(q_text, self.bm25_topn)
        except Exception:
            bm25 = []  # if rank-bm25 not installed, silently skip
        if not bm25:
            return dense
        if self.hybrid_fusion == "weighted":
            # Min-max normalize each list, then weighted sum by URL key
            def mm_norm(lst):
                if not lst:
                    return []
                import numpy as _np
                scores = _np.array([s for s, _ in lst], dtype="float32")
                mn, mx = float(scores.min()), float(scores.max())
                denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
                return [((s - mn) / denom, ch) for s, ch in lst]
            d = mm_norm(dense)
            b = mm_norm(bm25)
            def key(ch): return ch.get("url", "") + "::" + ch.get("title", "")
            agg: Dict[str, float] = {}
            ref: Dict[str, Dict] = {}
            for s, ch in d:
                kkey = key(ch); ref[kkey] = ch
                agg[kkey] = agg.get(kkey, 0.0) + (1.0 - self.bm25_weight) * float(s)
            for s, ch in b:
                kkey = key(ch); ref[kkey] = ch
                agg[kkey] = agg.get(kkey, 0.0) + self.bm25_weight * float(s)
            fused = sorted([(sc, ref[k]) for k, sc in agg.items()], key=lambda x: x[0], reverse=True)
        else:
            fused = _rrf_fuse(dense, bm25)  # default
        return fused[:k]

    # ----------------- public API -----------------
    @record_time("retrieval_ms")
    def retrieve(self, question: str, k: int | None = None, diversify: bool | None = None):
        k = k or self.top_k
        if diversify is None:
            diversify = True
        q_text = _expand_acronyms(question)

        # candidates
        cand_k = k * self.candidate_multiplier
        if self.hybrid_enabled:
            raw_hits = self._hybrid_candidates(q_text, cand_k)
        else:
            raw_hits = self._dense_candidates(q_text, cand_k)

        # optional rerank
        if self.use_rerank:
            self._maybe_init_reranker()
            raw_hits = self._rerank(q_text, raw_hits)

        if diversify:
            return _diversify_chunks(raw_hits, k=k, max_per_url=self.max_chunks_per_url)
        else:
            return [ch for _s, ch in raw_hits][:k]

    @record_time("end_to_end_ms")
    def answer(self, question: str):
        q_text = _expand_acronyms(question)

        # memory first
        mem_snips = self.mem_manager.retrieve(question, top_k=self.mem_k) if self.mem_enabled else []

        # candidates
        cand_k = self.top_k * self.candidate_multiplier
        if self.hybrid_enabled:
            raw_hits = self._hybrid_candidates(q_text, cand_k)
        else:
            raw_hits = self._dense_candidates(q_text, cand_k)

        if self.use_rerank:
            self._maybe_init_reranker()
            raw_hits = self._rerank(q_text, raw_hits)

        # select context
        context_chunks = _diversify_chunks(raw_hits, k=self.top_k, max_per_url=self.max_chunks_per_url)

        # gating (skip if rerank on)
        top_score = None
        if raw_hits:
            s0 = raw_hits[0][0]
            if isinstance(s0, (int, float)):
                top_score = float(s0)

        def _score_passes(score, threshold):
            if self.use_rerank:
                return True
            if score is None or threshold <= 0.0:
                return True
            if 0.0 <= score <= 1.0:
                return score >= threshold
            return True

        have_enough_hits = len([c for c in context_chunks if c.get("url", "")]) >= self.min_hits_for_grounded
        score_ok = _score_passes(top_score, self.score_threshold)

        if have_enough_hits and score_ok:
            if mem_snips:
                mem_chunk = {"title": "User memory", "url": "", "text": "• " + "\n• ".join(mem_snips)}
                context_chunks = [mem_chunk] + context_chunks

            prompt = build_prompt(question, context_chunks)
            answer = self._call_llm(prompt, system=SYSTEM)

            num_cites = max(1, int(self.citations_k))
            cite_urls = _select_citations(q_text, context_chunks, raw_hits, num_cites)

            if cite_urls and not re.search(r"(?im)^\s*Sources?\s*:", answer):
                footer = "Sources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(cite_urls))
                answer = f"{answer}\n\n{footer}"
            return answer

        if self.open_domain_fallback:
            mem_block = self._format_mem_block(mem_snips)
            general_prompt = (mem_block + f"User question: {question}").strip()
            return self._call_llm(general_prompt, system=GENERAL_SYSTEM)

        return "I couldn’t find enough information in the indexed corpus to answer that."
