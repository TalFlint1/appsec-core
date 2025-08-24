#!/usr/bin/env python
import os, re, csv, json, argparse
from pathlib import Path
import sys

# Ensure local package importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline  # noqa


def ensure_eval_seed(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["What is SQL Injection and one key mitigation?", r"cheatsheetseries.*SQL_Injection_Prevention"],
        ["Define SSRF and a common defense.", r"owasp\.org/Top10.*SSRF"],
        ["List the three main XSS types and a core mitigation.", r"cheatsheetseries.*XSS_Prevention"],
        ["What is a broken access control example?", r"owasp\.org/Top10.*Broken_Access_Control"],
        ["Explain insecure deserialization risk.", r"(owasp\.org/Top10|cheatsheetseries).*Deserialization"],
        ["What is command injection vs SQLi?", r"portswigger\.net.*(command-injection|SQL-injection)"],
        ["What is CSRF and a modern defense?", r"(cheatsheetseries.*Cross-Site_Request_Forgery|CSRF)"],
        ["Explain security misconfiguration and a hardening approach.", r"owasp\.org/Top10.*Security_Misconfiguration"],
        ["Give an example of cryptographic failure.", r"owasp\.org/Top10.*Cryptographic_Failures"],
        ["How to validate redirects/forwards safely?", r"cheatsheetseries.*Open_Redirect"],
        ["What is IDOR and how to prevent it?", r"portswigger\.net.*(idor|insecure-direct-object)"],
        ["Rate limiting best practices?", r"(cheatsheetseries.*Rate_Limiting|ASVS)"],
        ["What is path traversal and one mitigation?", r"portswigger\.net.*path-traversal"],
        ["Explain parameterized queries and why they matter.", r"cheatsheetseries.*SQL_Injection_Prevention"],
        ["CSP basics for XSS mitigation?", r"cheatsheetseries.*Content_Security_Policy"],
        ["What’s a secure password storage approach?", r"cheatsheetseries.*Password_Storage"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "expected_url_regex"])
        w.writerows(rows)


def ndcg_at_k(rank: int, k: int) -> float:
    """Single relevant doc DCG@k with gain=1."""
    if rank < 1 or rank > k:
        return 0.0
    import math
    return 1.0 / math.log2(rank + 1)


def run_eval(k: int, ablation: str, outdir: Path):
    pipe = RAGPipeline()
    diversify = (ablation != "baseline")  # baseline: diversification OFF
    eval_path = Path("eval") / "eval.csv"
    ensure_eval_seed(eval_path)

    total = 0
    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    details = []
    with eval_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            q = row["question"].strip()
            pat = row["expected_url_regex"].strip()
            try:
                chunks = pipe.retrieve(q, k=k, diversify=diversify)
                urls = [c.get("url", "") for c in chunks if c.get("url")]
                # find rank (1-indexed) of first match
                rank = 0
                for i, u in enumerate(urls, start=1):
                    if re.search(pat, u, re.IGNORECASE):
                        rank = i
                        break
                hit = (rank > 0 and rank <= k)
                details.append({"q": q, "pattern": pat, "urls": urls, "hit": bool(hit), "rank": rank or None})
                total += 1
                if hit:
                    hits += 1
                    mrr_sum += 1.0 / rank
                    ndcg_sum += ndcg_at_k(rank, k)
            except Exception as e:
                details.append({"q": q, "pattern": pat, "error": str(e), "hit": False})
                total += 1

    hit_rate = 100.0 * hits / max(1, total)
    recall_at_k = hit_rate  # with 1 relevant per query, recall@k == hit@k
    mrr_at_k = 100.0 * (mrr_sum / max(1, total))
    ndcg_at_k_pct = 100.0 * (ndcg_sum / max(1, total))

    out = {
        "top_k": k,
        "ablation": ablation,
        "total": total,
        "hits": hits,
        "hit_rate_pct": round(hit_rate, 2),
        "recall_at_k_pct": round(recall_at_k, 2),
        "mrr_at_k_pct": round(mrr_at_k, 2),
        "ndcg_at_k_pct": round(ndcg_at_k_pct, 2),
        "details": details[:50],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"hit_rate_{ablation}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append to metrics report
    md = [
        f"## Retrieval quality (ablation = {ablation})",
        f"- Top-{k} Hit@{k}: **{out['hit_rate_pct']}%** ({hits}/{total})",
        f"- Recall@{k}: **{out['recall_at_k_pct']}%**",
        f"- MRR@{k}: **{out['mrr_at_k_pct']}%**",
        f"- nDCG@{k}: **{out['ndcg_at_k_pct']}%**",
        "",
    ]
    report = outdir / "metrics_report.md"
    if not report.exists():
        report.write_text("# Metrics Report\n\n", encoding="utf-8")
    with report.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ablation", choices=["baseline", "current"], default="current")
    ap.add_argument("--outdir", default="metrics")
    args = ap.parse_args()
    run_eval(args.k, args.ablation, Path(args.outdir))


if __name__ == "__main__":
    main()
