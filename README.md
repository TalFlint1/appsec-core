# AppSec Core — OWASP Top‑10 Assistant

**AppSec Core** is a security‑focused Retrieval‑Augmented Generation (RAG) assistant built around OWASP Top‑10 content.
It’s designed to be **grounded, explainable, and measurable**: every answer cites sources; retrieval and end‑to‑end latencies are instrumented, and simple evaluation harnesses give you CV‑ready metrics.

---

## ✨ Key Features

- **RAG over a curated corpus** (OWASP Top‑10 2021, OWASP Cheat Sheets, PortSwigger, NIST…).  
  Index is FAISS, semantic encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- **Config‑driven retrieval**: top‑k, candidate expansion, per‑URL diversification, and optional **Cross‑Encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Acronym expansion** (SSRF/XSS/CSRF/IDOR/SQLi/SSTI → expanded) to stabilize retrieval.
- **Grounding gate** with similarity threshold + min hits; **open‑domain fallback** when RAG confidence is low.
- **Memory**:
  - Vector memory store.
  - Optional **LLM‑based memory extractor** (captures durable user facts only).
  - Streamlit sidebar to **enable/disable memory, add/list/clear**, and `/remember …` chat command.
- **Transparent citations** (configurable how many to show).
- **Instrumentation**: lightweight metrics that print **P50/P95** for retrieval and end‑to‑end.
- **Evaluation harnesses**: corpus stats, **Top‑K hit‑rate**, hallucination spot‑check CSVs, and memory A/B.

---

## 🧭 Repository Layout (high‑level)

```
configs/
  models.yaml              # LLM + embeddings + retrieval knobs
  sources.yaml             # crawl targets for the web connector
data/
  raw/crawl.jsonl          # scraped pages (URL + text)
  index/                   # FAISS index + metadata
metrics/                   # generated metrics & reports
scripts/
  build_index.py
  corpus_stats.py
  eval_hit_rate.py
  eval_hallucination.py
  eval_memory_ab.py
src/
  connectors/web           # simple web connector (crawler)
  index/faiss_store.py     # FAISS wrapper
  memory/store.py          # MemoryStore (vector)
  memory/manager.py        # LLM-based MemoryManager
  rag/pipeline.py          # RAGPipeline (retrieval, memory, LLM, gating)
  rag/prompts.py           # SYSTEM/GROUNDED prompts
  ui/streamlit_app.py      # Streamlit chat UI + Memory sidebar
  utils/metrics.py         # @record_time decorator & summary dumper
```

---

## 🚀 Quickstart (Local)

> Requires Python 3.10+

1) **Install deps**
```bash
pip install -r requirements.txt
```

2) **Configure models** (edit `configs/models.yaml`). Example:
```yaml
llm:
  provider: openai          # openai | ollama
  model: gpt-4o-mini
  temperature: 0.2

embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu               # or "cuda"

retrieval:
  top_k: 5
  candidate_multiplier: 3
  max_chunks_per_url: 2
  citations_k: 3
  # Confidence gate
  min_hits_for_grounded: 2
  score_threshold: 0.35
  open_domain_fallback: true
  # Memory
  mem_enabled: true
  mem_k: 3
  # (Optional) Reranker
  use_rerank: false
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_top_n: 30
```

3) **Set API key (OpenAI)** if using `provider: openai`
```bash
export OPENAI_API_KEY=sk-...           # macOS/Linux
# or set it in your shell/IDE/Colab environment
```

4) **Crawl sources**
```bash
python -m src.connectors.web --config configs/sources.yaml --out data/raw/crawl.jsonl
```

5) **Build index**
```bash
PYTHONPATH=. python scripts/build_index.py \
  --raw data/raw/crawl.jsonl \
  --index_dir data/index \
  --models_cfg configs/models.yaml
```

6) **Run the UI**
```bash
streamlit run src/ui/streamlit_app.py
```
- Use the **Memory** section in the sidebar (enable memory, add/list/clear).  
- Chat command: `/remember I live in Tel Aviv`

---

## 📏 Metrics & Evaluation (CV‑ready numbers)

These commands generate machine‑readable JSON and a human report at `metrics/metrics_report.md`.

```bash
# 0) (optional) clean
rm -rf metrics && mkdir -p metrics

# 1) Corpus stats
python scripts/corpus_stats.py

# 2) Retrieval quality (Top‑K hit rate)
python scripts/eval_hit_rate.py --k 5 --ablation baseline
python scripts/eval_hit_rate.py --k 5 --ablation current

# 3) Hallucination spot‑check (produces CSVs to manually label)
python scripts/eval_hallucination.py --gating on
python scripts/eval_hallucination.py --gating off

# 4) Memory A/B (optional)
python scripts/eval_memory_ab.py
```

**Artifacts produced** (under `metrics/`):
- `corpus_stats.json` — crawl size, domains, chunks, FAISS size/ntotal
- `hit_rate_baseline.json`, `hit_rate_current.json` — Top‑K hit‑rate
- `hallu_raw_on.csv`, `hallu_raw_off.csv` — answers + citations for manual labeling
- `latency_summary.json` — P50/P95 for retrieval and end‑to‑end
- `metrics_report.md` — single report aggregating everything

> **Hit‑rate definition:** for each eval question, if any of the retrieved URLs matches the expected regex, it’s a hit. This measures **retrieval alignment** with ground‑truth pages (not full answer quality).

---

## 🧠 Memory

- **Manual memory:** `/remember <fact>` inside chat, or use the Streamlit sidebar.
- **Auto extraction:** `MemoryManager` can extract durable, non‑sensitive facts from chat turns. Extraction is conservative (no PII, no guessing).
- **Retrieval:** Top memory snippets are appended (non‑cited) to the prompt for personalization.

> Memory is vectorized and ranked with recency/importance decay. You can toggle memory on/off in the UI and set `mem_k` in `configs/models.yaml`.

---

## 🧪 Reranking (optional)

Enable a cross‑encoder reranker for better Top‑K quality on short chunks:
```yaml
retrieval:
  use_rerank: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_top_n: 30
```
When rerank is ON, similarity **score gating is bypassed** (cross‑encoder scores aren’t cosine similarities).

---

## 📈 Interpreting Results

- **Hit‑rate up** (baseline → current) generally means corpus + retrieval knobs are improving alignment.  
- **Retrieval P95** reflects FAISS + embedding + rerank cost.  
- **End‑to‑end P95** includes model latency (LLM) and prompt size.  
- **Hallucinations**: after you label `hallu_raw_*.csv`, re‑summarize to compute `% unsupported claims` with and without confidence gating.

---

## 🛠️ Tips & Troubleshooting

- **Low hit‑rate?**
  - Increase `candidate_multiplier` (e.g., 5–8) and test.
  - Turn ON `use_rerank`.
  - Ensure acronym expansion is active (it is in `pipeline.py`).
  - Expand corpus (e.g., add OWASP Top‑10 2017 pages like XSS for better topical coverage) without hardcoding answers.
- **CUDA / cuDNN warnings in Colab:** benign if you’re on CPU; embeddings may load TF deps that warn. You can ignore these unless you specifically require GPU.
- **FAISS ntotal mismatch:** rebuild index with `scripts/build_index.py` after changing chunking or corpus.
- **OpenAI quota / key issues:** verify `OPENAI_API_KEY` and model name in `configs/models.yaml`.

---

## 🔧 Config Reference (excerpt)

```yaml
llm:
  provider: openai            # or ollama
  model: gpt-4o-mini
  temperature: 0.2

embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu                 # or cuda

retrieval:
  top_k: 5
  candidate_multiplier: 3
  max_chunks_per_url: 2
  citations_k: 3

  # Confidence gate for grounded mode
  min_hits_for_grounded: 2
  score_threshold: 0.35
  open_domain_fallback: true

  # Memory
  mem_enabled: true
  mem_k: 3

  # Optional reranker
  use_rerank: false
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_top_n: 30
```

---

## 📜 License

MIT.

---

## 🙌 Acknowledgements

- OWASP Top‑10 & Cheat Sheet Series  
- Sentence‑Transformers, FAISS, Streamlit, OpenAI
