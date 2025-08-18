# src/utils/metrics.py
from __future__ import annotations
import atexit, os, json, time, threading
from statistics import median

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, list[float]] = {}  # bucket -> list of durations (ms)
_OUTDIR = "metrics"
os.makedirs(_OUTDIR, exist_ok=True)

def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])

def record_time(bucket: str):
    """
    Decorator to record elapsed wall time (ms) per call in the given bucket.
    Summary is printed and written to metrics/latency_summary.json at process exit.
    """
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                with _METRICS_LOCK:
                    _METRICS.setdefault(bucket, []).append(dt_ms)
        return _wrapped
    return _decorator

def _flush():
    summary = {}
    with _METRICS_LOCK:
        for bucket, vals in _METRICS.items():
            if not vals:
                continue
            summary[bucket] = {
                "n": len(vals),
                "p50_ms": round(median(vals), 2),
                "p95_ms": round(_p95(vals), 2),
                "max_ms": round(max(vals), 2),
            }
    path = os.path.join(_OUTDIR, "latency_summary.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if summary:
        print("\n[latency] summary")
        for k, v in summary.items():
            print(f"- {k}: n={v['n']} p50={v['p50_ms']}ms p95={v['p95_ms']}ms max={v['max_ms']}ms")

# ensure it runs once at process exit
atexit.register(_flush)
