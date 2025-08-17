# src/rag/pipeline.py
import yaml, requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .prompts import build_prompt, SYSTEM, GENERAL_SYSTEM
from ..index.faiss_store import FaissStore
from ..memory.store import MemoryStore
import re
from collections import defaultdict

def _unique_urls_in_order(chunks):
    seen, out = set(), []
    for ch in chunks:
        u = ch.get("url", "")
        if not u or not u.startswith("http"):
            continue
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _diversify_chunks(hits, k, max_per_url):
    """hits: list of (score, chunk) best->worst; keep at most max_per_url per URL."""
    out, counts = [], defaultdict(int)
    for _score, ch in hits:
        url = ch.get("url","")
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

class RAGPipeline:
    def __init__(self, models_cfg="configs/models.yaml", index_dir="data/index"):
        self.cfg = load_yaml(models_cfg)
        self.index_dir = Path(index_dir)
        self.store = FaissStore(self.index_dir/"index.faiss", self.index_dir/"meta.jsonl")
        self.store.load()

        emb_model_name = self.cfg["embeddings"]["model_name"]
        emb_device = self.cfg["embeddings"].get("device", "cpu")
        self.embedder = SentenceTransformer(emb_model_name, device=emb_device)

        self.mem = MemoryStore(embedder=self.embedder)
        mem_cfg = (self.cfg.get("retrieval") or {})
        self.mem_enabled = bool(mem_cfg.get("mem_enabled", True))
        self.mem_k = int(mem_cfg.get("mem_k", 3))

        # LLM
        self.llm_provider = self.cfg["llm"]["provider"]
        self.llm_model    = self.cfg["llm"]["model"]
        self.temperature  = float(self.cfg["llm"].get("temperature", 0.2))

        # Retrieval knobs
        r = self.cfg.get("retrieval", {})
        self.top_k               = int(r.get("top_k", 5))
        self.candidate_multiplier= int(r.get("candidate_multiplier", 3))
        self.max_chunks_per_url  = int(r.get("max_chunks_per_url", 2))
        self.citations_k         = int(r.get("citations_k", 3))

        # Minimal hybrid gating
        self.min_hits_for_grounded = int(r.get("min_hits_for_grounded", 2))
        self.score_threshold       = float(r.get("score_threshold", 0.35))
        self.open_domain_fallback  = bool(r.get("open_domain_fallback", True))

    def _embed_query(self, q: str) -> np.ndarray:
        e = self.embedder.encode([q], normalize_embeddings=True)
        return np.asarray(e, dtype="float32")

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
            return resp.json().get("response","").strip()

        elif self.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            chat = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"system","content": system},
                          {"role":"user","content": prompt}],
                temperature=self.temperature,
            )
            return chat.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown llm provider: {self.llm_provider}")

    def answer(self, question: str):
        # 1) embed and retrieve more candidates than we'll use
        q_emb = self._embed_query(question)
        raw_hits = self.store.search(q_emb, top_k=self.top_k * self.candidate_multiplier)  # [(score, chunk), ...]

        # 2) diversify selected context
        context_chunks = _diversify_chunks(raw_hits, k=self.top_k, max_per_url=self.max_chunks_per_url)

        # 2.5) (NEW) retrieve relevant memories; prepend as pseudo-chunk(s)
        if self.mem_enabled:
            mem_snips = self.mem.search(question, top_k=self.mem_k)
            if mem_snips:
                mem_text = "• " + "\n• ".join(mem_snips)
                mem_chunk = {"title": "User memory", "url": "", "text": mem_text}
                # Prepend so it gets seen by the LLM but won’t be cited
                context_chunks = [mem_chunk] + context_chunks   

        # 3) compute retrieval confidence from the top score if it looks like a similarity
        top_score = None
        if raw_hits:
            s0 = raw_hits[0][0]
            if isinstance(s0, (int, float)):
                top_score = float(s0)

        def _score_passes(score, threshold):
            # Only enforce threshold if score appears to be a cosine-like similarity in [0,1].
            if score is None or threshold <= 0.0:
                return True
            if 0.0 <= score <= 1.0:
                return score >= threshold
            # Unknown scale (e.g., distances) → don't block on score
            return True

        have_enough_hits = len(context_chunks) >= self.min_hits_for_grounded
        score_ok = _score_passes(top_score, self.score_threshold)

        if have_enough_hits and score_ok:
            # 4) grounded answer from context
            prompt = build_prompt(question, context_chunks)
            answer = self._call_llm(prompt, system=SYSTEM)

            # citations ONLY from URLs we actually used
            cite_urls = _unique_urls_in_order(context_chunks)[: self.citations_k]
            if cite_urls and not re.search(r"(?im)^\s*Sources?\s*:", answer):
                footer = "Sources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(cite_urls))
                answer = f"{answer}\n\n{footer}"
            return answer

        # 5) fallback to general LLM (no citations)
        if self.open_domain_fallback:
            return self._call_llm(question, system=GENERAL_SYSTEM)

        # 6) strict mode (no fallback)
        return "I couldn’t find enough information in the indexed corpus to answer that."
