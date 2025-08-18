#!/usr/bin/env python
import os, csv, argparse, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline  # noqa

def ensure_seed(path: Path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [
        "Give a concrete SSRF example and one mitigation.",
        "What is the difference between authentication and authorization?",
        "Explain broken access control with an example.",
        "How does CSP help with XSS?",
        "What is SQL injection and how to prevent it?",
        "What is CSRF and how to defend against it?",
        "Describe security misconfiguration and a practical hardening step.",
        "What is IDOR?",
        "Explain parameterized queries and why they matter.",
        "What is the OWASP Top 10?",
    ]
    path.write_text("\n".join(seeds) + "\n", encoding="utf-8")

def run(gating: str, outdir: Path):
    pipe = RAGPipeline()
    # Gating ON = use current settings; OFF = disable retrieval thresholding
    if gating == "off":
        # make it easier to answer without strong retrieval (more likely to hallucinate)
        pipe.min_hits_for_grounded = 0
        pipe.score_threshold = 0.0

    qpath = Path("eval") / "hallu_questions.txt"
    ensure_seed(qpath)
    rows = []
    for q in qpath.read_text(encoding="utf-8").splitlines():
        q = q.strip()
        if not q: continue
        ans = pipe.answer(q)
        # extract cited URLs (simple footer scan)
        cites = []
        m = re.search(r"(?is)\nSources:\n(.*)$", ans)
        if m:
            for line in m.group(1).strip().splitlines():
                u = line.split("]",1)[-1].strip()  # crude, but fine for our footer "[n] url"
                u = u.lstrip("[] ").strip()
                cites.append(u)
        rows.append({"gating": gating, "question": q, "answer": ans, "citations": " | ".join(cites)})

    outdir.mkdir(parents=True, exist_ok=True)
    outcsv = outdir / f"hallu_raw_{gating}.csv"
    with outcsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gating","question","answer","citations"])
        w.writeheader(); w.writerows(rows)

    print(f"Wrote {outcsv}. Now manually label a copy with supported=true/false to compute % unsupported.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gating", choices=["on","off"], default="on")
    ap.add_argument("--outdir", default="metrics")
    args = ap.parse_args()
    run(args.gating, Path(args.outdir))
