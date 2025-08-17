# src/rag/pipeline.py
import yaml, json, requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .prompts import build_prompt, SYSTEM, GENERAL_SYSTEM
from ..index.faiss_store import FaissStore
import re
from collections import defaultdict
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def _unique_urls_in_order(chunks):
    seen, out = set(), []
    for ch in chunks:
        u = ch.get("url", "")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
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
        self.llm_provider = self.cfg["llm"]["provider"]
        self.llm_model = self.cfg["llm"]["model"]
        self.temperature = float(self.cfg["llm"].get("temperature", 0.2))

        # Retrieval knobs
        r = self.cfg.get("retrieval", {})
        self.top_k = int(r.get("top_k", 5))
        self.citations_k = int(r.get("citations_k", 3))
        self.candidate_multiplier = int(r.get("candidate_multiplier", 3))
        self.max_chunks_per_url = int(r.get("max_chunks_per_url", 2))

        # NEW: hybrid / fallback knobs
        self.min_hits_for_grounded = int(r.get("min_hits_for_grounded", 1))
        self.score_threshold = float(r.get("score_threshold", 0.0))
        self.open_domain_fallback = bool(r.get("open_domain_fallback", True))
        self.fallback_scope = r.get("fallback_scope", "any")  # "any" | "tech" | "off"
        self.date_time_aware = bool(r.get("date_time_aware", True))
        self.tz_name = r.get("timezone", "Asia/Jerusalem")

    def _embed_query(self, q: str) -> np.ndarray:
        e = self.embedder.encode([q], normalize_embeddings=True)
        return np.asarray(e, dtype="float32")

    def _call_llm(self, prompt: str, system: str = SYSTEM) -> str:
        if self.llm_provider == "ollama":
            # Use system + prompt for Ollama's /generate
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
                messages=[{"role":"system","content": system},{"role":"user","content": prompt}],
                temperature=self.temperature,
            )
            return chat.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unknown llm provider: {self.llm_provider}")

    # --- tiny utility: local date/time for "what day is it?" ---
    def _looks_like_date_question(self, q: str) -> bool:
        ql = q.lower()
        return any(p in ql for p in [
            "what day is it", "what's the day", "what is the date",
            "today's date", "what day today", "what date today"
        ])

    def _now_str(self):
        if not self.date_time_aware:
            return None
        try:
            tz = ZoneInfo(self.tz_name) if ZoneInfo else None
        except Exception:
            tz = None
        now = datetime.now(tz) if tz else datetime.now()
        return now.strftime("%A, %B %d, %Y")

    def _is_techy(self, q: str) -> bool:
        ql = q.lower()
        keys = [
            "api","docker","kubernetes","linux","windows","macos","python","java","javascript",
            "security","owasp","sql","xss","csrf","ssrf","encryption","authentication","authorization",
            "cloud","aws","gcp","azure","network","tls","ssh"
        ]
        return any(k in ql for k in keys)

    def answer(self, question: str, allow_open_fallback: bool | None = None) -> str:
        # 0) small utility path: date/time
        if self._looks_like_date_question(question):
            now = self._now_str()
            if now:
                return f"🟢 Mode: general\n\nToday is {now}."

        # 1) embed + retrieve
        q_emb = self._embed_query(question)
        raw_hits = self.store.search(q_emb, top_k=self.top_k * self.candidate_multiplier)

        # 2) diversify and assemble context
        context_chunks = _diversify_chunks(raw_hits, k=self.top_k, max_per_url=self.max_chunks_per_url)

        # Determine retrieval strength
        top_score = None
        if raw_hits:
            s0 = raw_hits[0][0]
            if isinstance(s0, (int, float)):
                top_score = float(s0)
        have_ground = len(context_chunks) >= self.min_hits_for_grounded and (
            top_score is None or top_score >= self.score_threshold
        )

        # 3) grounded answer with citations
        if have_ground:
            prompt = build_prompt(question, context_chunks)
            answer = self._call_llm(prompt, system=SYSTEM)

            # citations ONLY from used chunks
            cite_urls = _unique_urls_in_order(context_chunks)[: self.citations_k]
            if cite_urls and not re.search(r"(?im)^\s*Sources?\s*:", answer):
                footer = "Sources:\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(cite_urls))
                answer = f"{answer}\n\n{footer}"

            return f"🟢 Mode: grounded\n\n{answer}"

        # 4) fallback if allowed
        if allow_open_fallback is None:
            allow_open_fallback = self.open_domain_fallback

        if allow_open_fallback and self.fallback_scope != "off":
            if self.fallback_scope == "tech" and not self._is_techy(question):
                return "🔒 Mode: domain-only\n\nI’m optimized for security/OWASP. Rephrase with a security angle, or enable general fallback."

            out = self._call_llm(question, system=GENERAL_SYSTEM)
            return f"🟢 Mode: general\n\n{out}"

        # 5) strict mode
        return "🔒 Mode: domain-only\n\nI couldn’t find enough information in the indexed corpus to answer that."