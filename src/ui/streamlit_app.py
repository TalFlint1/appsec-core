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

# Sidebar: corpus + controls
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

    # ------- Memory controls -------
    st.divider()
    st.header("Memory")
    mem_enabled_default = getattr(pipe, "mem_enabled", True)
    mem_enabled_ui = st.checkbox("Enable memory", value=mem_enabled_default)
    pipe.mem_enabled = mem_enabled_ui  # reflect UI state into pipeline

    # Auto-capture toggle (LLM-based)
    auto_capture = st.checkbox(
        "Auto-capture from messages",
        value=True,
        help="Extract durable, non-sensitive facts via the LLM and store them."
    )

    new_mem = st.text_input("Add a memory (e.g., “I live in Tel Aviv”).", key="mem_add")
    if st.button("Save memory"):
        if new_mem.strip():
            try:
                pipe.mem.add(new_mem.strip())
                st.success("Saved to memory.")
            except Exception as e:
                st.error(f"Could not save memory: {e}")
        else:
            st.warning("Write something first.")

    if st.button("Show memory items"):
        try:
            items = pipe.mem.all()
            if not items:
                st.info("No memory yet.")
            else:
                for it in items[-20:]:
                    st.write("•", it["text"])
        except Exception as e:
            st.error(f"Could not read memory: {e}")

    if st.button("Clear memory"):
        try:
            pipe.mem.clear()
            st.warning("Memory cleared.")
        except Exception as e:
            st.error(f"Could not clear memory: {e}")

    # Dev helper: show what memory would be retrieved for the CURRENT input (when provided)
    show_mem_debug = st.checkbox("Show retrieved memory for current question (dev)", value=False)
    # ------- end Memory controls -------

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
    # ---- Chat command: /remember <text> ----
    low = prompt.strip().lower()
    if low.startswith("/remember "):
        fact = prompt.strip()[10:].strip()
        if fact:
            try:
                pipe.mem.add(fact)
                st.session_state.chat.append(("assistant", "Saved to memory."))
            except Exception as e:
                st.session_state.chat.append(("assistant", f"Could not save memory: {e}"))
        else:
            st.session_state.chat.append(("assistant", "Nothing to remember."))
        st.stop()
    # ---- end command ----

    # Render user message immediately
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- NEW: auto-learn via MemoryManager (shared store with /remember) ----
    try:
        if getattr(pipe, "mem_enabled", True) and auto_capture:
            recent = st.session_state.chat[-6:]  # last few turns, (role, msg)
            saved = pipe.mem_manager.maybe_extract_and_store(prompt, recent_chat=recent, max_facts=3)
            if saved:
                st.sidebar.success("Saved memory: " + "; ".join(f"“{s.text}”" for s in saved))
    except Exception as e:
        st.sidebar.error(f"Memory extraction failed: {e}")
    # ------------------------------------------------------------------------

    # Optional: show what memory is retrieved for THIS prompt (debug)
    if show_mem_debug and getattr(pipe, "mem_enabled", True):
        try:
            snips = pipe.mem_manager.retrieve(prompt, top_k=getattr(pipe, "mem_k", 3))
            if snips:
                st.sidebar.info("Retrieved memory:\n" + "\n".join("• " + s for s in snips))
            else:
                st.sidebar.info("Retrieved memory: (none)")
        except Exception as e:
            st.sidebar.error(f"Memory retrieve failed: {e}")

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
