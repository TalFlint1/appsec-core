#!/usr/bin/env python
import os, json, re, argparse, time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline  # noqa

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # crude but consistent

def ensure_seed(path: Path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    demos = [
        {
            "turns": [
                {"role":"user","text":"I'm focusing on SSRF this week. Remember that."},
                {"role":"user","text":"Remind me—what should I check first when reviewing SSRF risk?"}
            ],
            "check_regex": r"SSRF",  # expect SSRF to be discussed
        },
        {
            "turns": [
                {"role":"user","text":"My favorite topic is SQL injection; store that."},
                {"role":"user","text":"Give me a quick checklist for preventing it."}
            ],
            "check_regex": r"parameteri?zed|prepared statement|bind",  # expect mitigation hints
        }
    ]
    path.write_text("\n".join(json.dumps(x) for x in demos) + "\n", encoding="utf-8")

def run_once(mem_enabled: bool, dialogs_path: Path):
    pipe = RAGPipeline()
    pipe.mem.clear()  # fresh run
    pipe.mem_enabled = mem_enabled
    results = []
    for line in dialogs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        turns = d["turns"]; check = d.get("check_regex","")
        agg_tokens = 0; agg_latency = 0.0; last_answer = ""
        for t in turns:
            if t["role"] != "user":
                continue
            t0 = time.perf_counter()
            ans = pipe.answer(t["text"])
            agg_latency += (time.perf_counter() - t0) * 1000.0
            agg_tokens += _approx_tokens(t["text"]) + _approx_tokens(ans)
            last_answer = ans
        ok = bool(re.search(check, last_answer, re.IGNORECASE)) if check else True
        results.append({"success1": ok, "tokens_used": agg_tokens, "end_to_end_ms": round(agg_latency,2)})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", default="eval/memory_dialogs.jsonl")
    ap.add_argument("--outdir", default="metrics")
    args = ap.parse_args()

    dpath = Path(args.dialogs); ensure_seed(dpath)
    off = run_once(mem_enabled=False, dialogs_path=dpath)
    on  = run_once(mem_enabled=True,  dialogs_path=dpath)

    def _avg(xs, key): 
        vals = [x[key] for x in xs]
        return sum(vals)/max(1,len(vals))
    def _rate(xs, key): 
        vals = [1 if x[key] else 0 for x in xs]
        return 100.0*sum(vals)/max(1,len(vals))

    summary = {
        "n_dialogs": len(on),
        "success1_off_pct": round(_rate(off, "success1"), 2),
        "success1_on_pct":  round(_rate(on,  "success1"), 2),
        "success1_delta_pct": round(_rate(on,"success1") - _rate(off,"success1"), 2),
        "token_off_avg": round(_avg(off, "tokens_used"), 1),
        "token_on_avg":  round(_avg(on,  "tokens_used"), 1),
        "token_delta_pct": round(100.0*(_avg(on,"tokens_used") - _avg(off,"tokens_used"))/max(1.0,_avg(off,"tokens_used")), 2),
        "latency_off_ms_avg": round(_avg(off, "end_to_end_ms"), 2),
        "latency_on_ms_avg":  round(_avg(on,  "end_to_end_ms"), 2),
        "latency_delta_pct":  round(100.0*(_avg(on,"end_to_end_ms") - _avg(off,"end_to_end_ms"))/max(1.0,_avg(off,"end_to_end_ms")), 2),
    }

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "memory_ab.json").write_text(json.dumps({"off":off,"on":on,"summary":summary}, indent=2), encoding="utf-8")

    # append to report
    md = [
        "## Memory A/B",
        f"- N dialogs: **{summary['n_dialogs']}**",
        f"- Success@1 delta: **{summary['success1_delta_pct']} pp** (off={summary['success1_off_pct']}%, on={summary['success1_on_pct']}%)",
        f"- Tokens avg delta: **{summary['token_delta_pct']}%** (off={summary['token_off_avg']}, on={summary['token_on_avg']})",
        f"- Latency avg delta: **{summary['latency_delta_pct']}%** (off={summary['latency_off_ms_avg']}ms, on={summary['latency_on_ms_avg']}ms)",
        "",
    ]
    report = outdir / "metrics_report.md"
    if not report.exists():
        report.write_text("# Metrics Report\n\n", encoding="utf-8")
    with report.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
