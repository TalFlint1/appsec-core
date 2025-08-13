# src/rag/pipeline.py
import yaml, json, requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .prompts import build_prompt, SYSTEM
from ..index.faiss_store import FaissStore
import re
from collections import defaultdict

def _diversify_chunks(hits, k, max_per_url):
    """
    hits: list of (score_or_dist, chunk) ordered best->worst.
    Returns up to k chunks with at most max_per_url per URL.
    """
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

def _unique_urls_in_order(chunks):
    """Return URLs in the order they appear, no duplicates."""
    seen, out = set(), []
    for ch in chunks:
        u = ch.get("url", "")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

def load_yaml(path): 
    with open(path, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def _is_locale_url(url: str) -> bool:
    # Matches /Top10/fr/... /Top10/de/... /Top10/ja/... etc.
    # Add common language codes + pt-br explicitly.
    return bool(re.search(r"/Top10/(?:[a-z]{2}|pt-br|zh|ja|ko)/", url, re.IGNORECASE))

class RAGPipeline:
    def __init__(self, models_cfg="configs/models.yaml", index_dir="data/index"):
        self.cfg = load_yaml(models_cfg)
        self.index_dir = Path(index_dir)
        self.store = FaissStore(self.index_dir/"index.faiss", self.index_dir/"meta.jsonl")
        self.store.load()
        emb_model_name = self.cfg["embeddings"]["model_name"]            
        # Initialise the embedding model on the configured device. Without
        # specifying device, SentenceTransformer defaults to CPU even on GPU
        # systems such as Colab.
        emb_device = self.cfg["embeddings"].get("device", "cpu")
        self.embedder = SentenceTransformer(emb_model_name, device=emb_device)
        self.top_k = int(self.cfg["retrieval"].get("top_k", 5))
        self.llm_provider = self.cfg["llm"]["provider"]
        self.llm_model = self.cfg["llm"]["model"]
        self.temperature = float(self.cfg["llm"].get("temperature", 0.2))

    def _embed_query(self, q: str) -> np.ndarray:
        e = self.embedder.encode([q], normalize_embeddings=True)
        return np.asarray(e, dtype="float32")

    def _call_llm(self, prompt: str) -> str:
        if self.llm_provider == "ollama":
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_model, "prompt": f"{SYSTEM}\n\n{prompt}", "stream": False, "options": {"temperature": self.temperature}},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response","").strip()

        elif self.llm_provider == "openai":
            from openai import OpenAI
            import os
            # Expect OPENAI_API_KEY in env (no code changes needed for keys)
            client = OpenAI()
            chat = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"system","content": SYSTEM},{"role":"user","content": prompt}],
                temperature=self.temperature,
            )
            return chat.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unknown llm provider: {self.llm_provider}")

    def answer(self, question: str):
        # 1) embed query
        q_emb = self._embed_query(question)

        # 2) read knobs
        top_k = int(self.cfg["retrieval"].get("top_k", 5))
        cand_mult = int(self.cfg["retrieval"].get("candidate_multiplier", 3))
        max_per_url = int(self.cfg["retrieval"].get("max_chunks_per_url", 2))
        citations_k = int(self.cfg["retrieval"].get("citations_k", 3))

        # 3) retrieve more candidates than we’ll use, to allow diversification
        raw_hits = self.store.search(q_emb, top_k=top_k * cand_mult)

        # 4) diversify context so it isn’t all from one URL
        context_chunks = _diversify_chunks(raw_hits, k=top_k, max_per_url=max_per_url)

        # 5) build prompt only from those chunks and ask the LLM
        prompt = build_prompt(question, context_chunks)
        answer = self._call_llm(prompt)

        # 6) citations come ONLY from the exact chunks we used (max citations_k)
        cite_urls = _unique_urls_in_order(context_chunks)[:citations_k]

        # 7) append a Sources footer iff we have any
        if cite_urls and not re.search(r"(?im)^\s*Sources?\s*:", answer):
            footer = "Sources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(cite_urls))
            answer = f"{answer}\n\n{footer}"

        return answer