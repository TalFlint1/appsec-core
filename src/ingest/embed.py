# src/ingest/embed.py
from typing import List, Dict, Tuple
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_records(raw_jsonl: str) -> List[Dict]:
    records = []
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def embed_chunks(chunks: List[Dict], model_name: str, device: str = "cpu") -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed a list of chunk dictionaries using a sentence-transformers model.

    Args:
        chunks: List of chunk dicts with a "text" field.
        model_name: Name or path of the sentence-transformers model.
        device: PyTorch device to load the model on ("cuda" or "cpu").

    Returns:
        A tuple of (embeddings array, list of chunk dicts). Embeddings are normalised.
    """
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype="float32"), chunks