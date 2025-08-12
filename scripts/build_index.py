# scripts/build_index.py
import argparse, json
from pathlib import Path
from tqdm import tqdm
import yaml
from src.ingest.chunkers import chunk_record
from src.ingest.embed import load_records, embed_chunks
from src.index.faiss_store import FaissStore

def load_models_cfg(path="configs/models.yaml"):
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(raw_jsonl, index_dir, models_cfg):
    cfg = load_models_cfg(models_cfg)
    recs = load_records(raw_jsonl)
    all_chunks = []
    for r in tqdm(recs, desc="chunking"):
        all_chunks.extend(chunk_record(r, cfg["retrieval"]["chunk_size"], cfg["retrieval"]["chunk_overlap"]))
    # Pass through the configured device (e.g. "cuda" on Colab) when embedding.
    embs, meta = embed_chunks(
        all_chunks,
        cfg["embeddings"]["model_name"],
        device=cfg["embeddings"].get("device", "cpu"),
    )
    store = FaissStore(Path(index_dir)/"index.faiss", Path(index_dir)/"meta.jsonl")
    store.build(embs, meta)
    store.save()
    print(f"[index] saved to {index_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/crawl.jsonl")
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--models_cfg", default="configs/models.yaml")
    args = ap.parse_args()
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    main(args.raw, args.index_dir, args.models_cfg)
