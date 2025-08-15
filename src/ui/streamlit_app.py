# src/ui/streamlit_app.py
import os, json, re
from pathlib import Path
import streamlit as st

# Make local package importable when running `streamlit run src/ui/streamlit_app.py`
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline  # noqa

st.set_page_config(page_title="DocPilot RAG", page_icon="🤖", layout="wide")

@st.cache_resource(show_spinner=True)
def load_pipe():
    return RAGPipeline()

pipe = load_pipe()

# Sidebar: corpus snapshot
with st.sidebar:
    st.header("Corpus")
    crawl_path = ROOT / "data" / "raw" / "crawl.jsonl"
    if crawl_path.exists():
        n_lines = sum(1 for _ in crawl_path.open("r", encoding="utf-8"))
        st.write(f"Pages indexed: **{n_lines}**")
        # quick domain mix
        try:
            import collections
            domains = []
            for i, line in enumerate(crawl_path.open("r", encoding="utf-8")):
                if i > 500: break
                url = json.loads(line).get("url","")
                m = re.match(r"https?://([^/]+)/", url)
                if m: domains.append(m.group(1).lower())
            cnt = collections.Counter(domains).most_common(6)
            if cnt:
                st.caption("Top domains (sample):")
                for d,c in cnt:
                    st.write(f"- {d} × {c}")
        except Exception:
            pass
    else:
        st.info("No crawl snapshot found yet.")

    st.divider()
    st.header("Answer style")
    length = st.radio("Length", ["Concise (3–5 sent.)", "Medium (6–9)", "Long (10–14)"], index=1)
    followup = st.checkbox("End with a short follow-up question", value=True)
    show_sources = st.checkbox("Show sources", value=True)
    max_sources = st.slider("Max sources", min_value=1, max_value=5, value=3)

st.title("DocPilot RAG")
st.caption("Ask anything about the OWASP Top-10 and related references.")

# Chat history (client-side only)
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Type your question…")
if prompt:
    # Render user message immediately
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build soft style hints for the LLM (no code changes needed)
    style_bits = []
    if length.startswith("Concise"):
        style_bits.append("Please answer in about 3–5 sentences.")
    elif length.startswith("Medium"):
        style_bits.append("Please answer in about 6–9 sentences.")
    else:
        style_bits.append("Please answer in about 10–14 sentences.")
    if followup:
        style_bits.append("End with one brief, relevant follow-up question.")
    style_hint = "\n\n" + " ".join(style_bits) if style_bits else ""

    # Ask the pipeline
    try:
        answer = pipe.answer(prompt + style_hint)
    except Exception as e:
        answer = f"**Error:** {e}"

    # Optionally trim sources footer
    if not show_sources:
        answer = re.sub(r"\n+Sources:\n(.|\n)*$", "", answer, flags=re.IGNORECASE)

    # Clip number of sources in the rendered footer (cosmetic)
    if show_sources and max_sources > 0:
        def clip_sources(md: str) -> str:
            m = re.search(r"(?is)\nSources:\n(.*)$", md)
            if not m: return md
            body = md[:m.start()]
            block = m.group(1).strip()
            lines = [ln for ln in block.splitlines() if ln.strip()]
            kept = lines[:max_sources]
            return body + "\n\nSources:\n" + "\n".join(kept)
        answer = clip_sources(answer)

    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
