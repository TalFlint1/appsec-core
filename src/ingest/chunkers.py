# src/ingest/chunkers.py
import re
from typing import List, Dict

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"])')

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def chunk(text: str, chunk_size_tokens=800, overlap_tokens=120) -> List[str]:
    # Approximate tokens as words
    sents = _split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        w = s.split()
        if cur_len + len(w) > chunk_size_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap by words
            if overlap_tokens > 0:
                overlap = " ".join(" ".join(cur).split()[-overlap_tokens:])
                cur = [overlap]
                cur_len = len(overlap.split())
            else:
                cur, cur_len = [], 0
        cur.append(s); cur_len += len(w)
    if cur: chunks.append(" ".join(cur))
    return chunks

def chunk_record(rec: Dict, chunk_size_tokens=800, overlap_tokens=120):
    return [
        {
            "id": f"{rec['url']}#chunk{idx}",
            "url": rec["url"],
            "title": rec.get("title",""),
            "text": ch
        }
        for idx, ch in enumerate(chunk(rec["text"], chunk_size_tokens, overlap_tokens))
    ]
