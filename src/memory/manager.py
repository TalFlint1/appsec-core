from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json, time, math

@dataclass
class MemoryFact:
    text: str
    category: str = "preference"   # e.g., "profile" | "preference" | "context"
    importance: int = 3            # 1..5
    ttl_days: int = 365            # how long to keep
    ts: float = 0.0                # unix seconds

class MemoryManager:
    """
    LLM-based extractor for semantic memory. Decides what to store, persists via MemoryStore,
    and retrieves relevant facts for a query.
    """
    def __init__(self, store, embedder, llm_provider: str, llm_model: str, temperature: float = 0.0):
        self.store = store
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature

    # ---- Extraction ----
    def maybe_extract_and_store(self, new_user_msg: str, recent_chat: List[tuple[str,str]], max_facts: int = 3) -> List[MemoryFact]:
        """
        Ask the LLM: are there durable, non-sensitive facts to remember?
        Return any stored facts.
        """
        convo_tail = "\n".join(f"{r.upper()}: {m}" for r, m in recent_chat[-6:])
        system = (
            "You extract stable, privacy-safe user facts from chat. "
            "Only include facts the user explicitly told you (no guessing). "
            "NEVER include secrets, exact addresses, phone, email, IDs, medical data, or anything sensitive. "
            "If nothing to remember, return an empty list."
        )
        user = {
            "conversation_tail": convo_tail,
            "new_user_message": new_user_msg,
            "instruction": (
                "Return JSON with key 'facts' as a list of objects: "
                "{text, category, importance (1-5), ttl_days}. "
                "Keep each 'text' short and self-contained. "
                f"Max {max_facts} items."
            ),
        }
        facts = self._call_llm_for_facts(system, user)
        stored: List[MemoryFact] = []
        for f in facts:
            mf = MemoryFact(
                text=f.get("text","").strip(),
                category=f.get("category","preference"),
                importance=int(f.get("importance",3)),
                ttl_days=int(f.get("ttl_days",365)),
                ts=time.time(),
            )
            if mf.text:
                self.store.add_with_meta(mf.__dict__)
                stored.append(mf)
        return stored

    def _call_llm_for_facts(self, system: str, user_payload: Dict[str,Any]) -> List[Dict[str,Any]]:
        if self.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content": system},
                    {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            try:
                data = json.loads(resp.choices[0].message.content)
                facts = data.get("facts", [])
                return facts if isinstance(facts, list) else []
            except Exception:
                return []
        elif self.llm_provider == "ollama":
            import requests
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_model,
                      "prompt": f"{system}\n\n{json.dumps(user_payload, ensure_ascii=False)}\nReturn only JSON with key 'facts'.",
                      "stream": False,
                      "options": {"temperature": self.temperature}},
                timeout=60,
            )
            try:
                data = json.loads(r.json().get("response","{}"))
                facts = data.get("facts", [])
                return facts if isinstance(facts, list) else []
            except Exception:
                return []
        else:
            return []

    # ---- Retrieval (semantic + recency/importance weighting) ----
    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        items = self.store.all()
        if not items:
            return []
        texts = [it["text"] for it in items]
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        import numpy as np
        sims = (embs @ q).tolist()

        now = time.time()
        scored = []
        for it, sim in zip(items, sims):
            age_days = max(0.0, (now - it.get("ts", now)) / 86400.0)
            imp = max(1, min(5, int(it.get("importance",3))))
            # exponential recency decay with half-life ~90d
            decay = 0.5 ** (age_days / 90.0)
            score = sim * (1 + 0.15*(imp-3)) * (0.6 + 0.4*decay)
            scored.append((score, it["text"]))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:top_k]]
