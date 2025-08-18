#!/usr/bin/env python
import os, json, re, argparse
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

def canonicalize(url: str) -> str:
    from urllib.parse import urlparse, urlunparse
    u = urlparse(url)
    # strip trailing "index.html" and duplicate slashes; avoid adding slash to *.ext
    path = re.sub(r"/index\.html?$", "/", u.path)
    path = re.sub(r"/{2,}", "/", path)
    if not path.endswith("/") and not re.search(r"\.[A-Za-z0-9]{1,5}$", path):
        path = path + "/"
    return urlunparse((u.scheme, u.netloc, path, u.params, u.query, ""))

def count_lines(p: Path) -> int:
    if not p.exists(): return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/crawl.jsonl")
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--outdir", default="metrics")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Raw corpus URLs (docs)
    raw_path = Path(args.raw)
    urls, domains = [], []
    if raw_path.exists():
        import json
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    u = rec.get("url","")
                    if u:
                        cu = canonicalize(u)
                        urls.append(cu)
                        domains.append(urlparse(cu).netloc.lower())
                except Exception:
                    pass
    unique_urls = sorted(set(urls))
    domain_hist = Counter(domains)

    # Chunks + FAISS
    meta_path  = Path(args.index_dir) / "meta.jsonl"
    faiss_path = Path(args.index_dir) / "index.faiss"
    chunk_count = count_lines(meta_path)
    faiss_filesize = faiss_path.stat().st_size if faiss_path.exists() else 0

    faiss_ntotal = 0
    try:
        import faiss
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            faiss_ntotal = int(index.ntotal)
    except Exception:
        pass

    stats = {
        "crawl_size_docs": len(unique_urls),
        "unique_domains": len(set(domains)),
        "top_domains": domain_hist.most_common(10),
        "chunk_count": chunk_count,
        "faiss_ntotal": faiss_ntotal,
        "faiss_filesize_bytes": faiss_filesize,
    }

    # Save JSON
    (outdir / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append/Write human-readable markdown
    md = [
        "## Corpus",
        f"- Crawl size (unique URLs): **{stats['crawl_size_docs']}**",
        f"- Unique domains: **{stats['unique_domains']}**",
        f"- Chunks: **{stats['chunk_count']}**",
        f"- FAISS ntotal: **{stats['faiss_ntotal']}**, index size: **{stats['faiss_filesize_bytes']/1_000_000:.2f} MB**",
        "",
        "| Domain | Count |",
        "|---|---:|",
    ]
    for d,c in stats["top_domains"]:
        md.append(f"| `{d}` | {c} |")
    md.append("")
    report = outdir / "metrics_report.md"
    if not report.exists():
        report.write_text("# Metrics Report\n\n", encoding="utf-8")
    with report.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
