# src/rag/pipeline.py
import re
import yaml
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from .prompts import build_prompt, SYSTEM, GENERAL_SYSTEM
from ..index.faiss_store import FaissStore
from ..memory.store import MemoryStore
from ..memory.manager import MemoryManager
from ..utils.metrics import record_time

def _unique_urls_in_order(chunks):
    seen, out = set(), []
    for ch in chunks:
        u = ch.get("url", "")
        if not u or not u.startswith("http"):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _diversify_chunks(hits, k, max_per_url):
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
        self.candidate_multiplier  = int(r.get("candidate_multiplier", 3))
        self.max_chunks_per_url    = int(r.get("max_chunks_per_url", 2))
        self.citations_k           = int(r.get("citations_k", 3))
        self.min_hits_for_grounded = int(r.get("min_hits_for_grounded", 2))
        self.score_threshold       = float(r.get("score_threshold", 0.35))
        self.open_domain_fallback  = bool(r.get("open_domain_fallback", True))

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
        self.mem         = self.mem_store  # keep UI compatibility
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
    def _expand_acronyms(self, q: str) -> str:
        def repl(m):
            t = m.group(0)
            exp = ACRONYM_MAP.get(t.lower())
            return f"{t} ({exp})" if exp else t
        return re.sub(r"\b(SSRF|XSS|CSRF|IDOR|SQLi|SSTI)\b", repl, q, flags=re.IGNORECASE)

    def _embed_query(self, q: str) -> np.ndarray:
        e = self.embedder.encode([q], normalize_embeddings=True)
        return np.asarray(e, dtype="float32")

    def _maybe_init_reranker(self):
        if not self.use_rerank:
            return
        if self.reranker is None:
            self.reranker = CrossEncoder(self.reranker_model, max_length=512)

    def _rerank(self, question: str, hits):
        """hits = list[(score, chunk)]. Return list[(new_score, chunk)] sorted desc."""
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

    # ----------------- public API -----------------
    @record_time("retrieval_ms")
    def retrieve(self, question: str, k: int | None = None, diversify: bool | None = None):
        """Return the top-k chunks for a question (optionally diversified)."""
        k = k or self.top_k
        if diversify is None:
            diversify = True
        q_text = self._expand_acronyms(question)
        q_emb = self._embed_query(q_text)

        raw_hits = self.store.search(q_emb, top_k=k * self.candidate_multiplier)  # [(score, chunk), ...]
        if self.use_rerank:
            self._maybe_init_reranker()
            raw_hits = self._rerank(q_text, raw_hits)

        if diversify:
            return _diversify_chunks(raw_hits, k=k, max_per_url=self.max_chunks_per_url)
        else:
            return [ch for _s, ch in raw_hits][:k]

    @record_time("end_to_end_ms")
    def answer(self, question: str):
        q_text = self._expand_acronyms(question)

        # 0) memory first (non-cited)
        mem_snips = self.mem_manager.retrieve(question, top_k=self.mem_k) if self.mem_enabled else []

        # 1) retrieval (with optional rerank)
        q_emb = self._embed_query(q_text)
        raw_hits = self.store.search(q_emb, top_k=self.top_k * self.candidate_multiplier)
        if self.use_rerank:
            self._maybe_init_reranker()
            raw_hits = self._rerank(q_text, raw_hits)

        # 2) select context
        context_chunks = _diversify_chunks(raw_hits, k=self.top_k, max_per_url=self.max_chunks_per_url)

        # 3) simple confidence gating
        top_score = None
        if raw_hits:
            s0 = raw_hits[0][0]
            if isinstance(s0, (int, float)):
                top_score = float(s0)

        def _score_passes(score, threshold):
            # If reranking is on, skip score gating (cross-encoder scores aren’t cosine).
            if self.use_rerank:
                return True
            if score is None or threshold <= 0.0:
                return True
            if 0.0 <= score <= 1.0:
                return score >= threshold
            return True

        have_enough_hits = len([c for c in context_chunks if c.get("url", "")]) >= self.min_hits_for_grounded
        score_ok = _score_passes(top_score, self.score_threshold)

        # 4) grounded mode
        if have_enough_hits and score_ok:
            if mem_snips:
                mem_chunk = {"title": "User memory", "url": "", "text": "• " + "\n• ".join(mem_snips)}
                context_chunks = [mem_chunk] + context_chunks

            prompt = build_prompt(question, context_chunks)
            answer = self._call_llm(prompt, system=SYSTEM)

            cite_urls = _unique_urls_in_order(context_chunks)[: self.citations_k]
            if cite_urls and not re.search(r"(?im)^\s*Sources?\s*:", answer):
                footer = "Sources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(cite_urls))
                answer = f"{answer}\n\n{footer}"
            return answer

        # 5) open-domain fallback (with a tiny memory preface)
        if self.open_domain_fallback:
            mem_block = self._format_mem_block(mem_snips)
            general_prompt = (mem_block + f"User question: {question}").strip()
            return self._call_llm(general_prompt, system=GENERAL_SYSTEM)

        # 6) strict mode
        return "I couldn’t find enough information in the indexed corpus to answer that."
