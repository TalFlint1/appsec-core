# src/rag/pipeline.py
import yaml, json, requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .prompts import build_prompt, SYSTEM
from ..index.faiss_store import FaissStore

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
        self.embedder = SentenceTransformer(emb_model_name)
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
        q_emb = self._embed_query(question)
        hits = self.store.search(q_emb, top_k=self.top_k)
        passages = [h[1] for h in hits]
        prompt = build_prompt(question, passages)
        answer = self._call_llm(prompt)
        # Build a clean Sources footer from passages
        sources = [p["url"] for p in passages]
        footer = "Sources:\n" + "\n".join([f"[{i+1}] {u}" for i,u in enumerate(sources)])
        return answer if answer.strip().endswith("Sources:") else f"{answer}\n\n{footer}"
