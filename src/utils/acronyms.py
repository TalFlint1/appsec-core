# src/utils/acronyms.py
import re

# Small, high-signal expansions only (generic, no URLs)
ACRONYM_MAP = {
    r"\bcsrf\b": ["cross-site request forgery", "cross site request forgery", "anti-forgery token", "csrf token", "samesite"],
    r"\bxss\b":  ["cross-site scripting", "cross site scripting", "content security policy", "csp"],
    r"\bssrf\b": ["server-side request forgery", "server side request forgery"],
    r"\bsqli\b": ["sql injection", "parameterized queries", "prepared statements", "query parameterization"],
    r"\bxxe\b":  ["xml external entity", "xml external entities"],
    r"\bidor\b": ["insecure direct object reference", "direct object reference"],
    r"\bcsp\b":  ["content security policy"],
    r"\bhsts\b": ["http strict transport security"],
    r"\bjwt\b":  ["json web token"],
}

def _terms_for_query(q: str) -> list[str]:
    ql = q.lower()
    terms = []
    for pat, syns in ACRONYM_MAP.items():
        if re.search(pat, ql) or any(s in ql for s in syns):
            terms.extend(syns)
    # normalize hyphen/space variants + dedupe
    norm = set()
    for t in terms:
        norm.add(t)
        norm.add(t.replace("-", " "))
        norm.add(t.replace(" ", "-"))
    return list(norm)

def expand_query_text(query: str, max_extra_chars: int = 300) -> str:
    """Append expansions so dense embedding captures both acronym and full name."""
    extra = " ".join(_terms_for_query(query))
    if not extra:
        return query
    extra = extra[:max_extra_chars]
    return f"{query} {extra}"

def acronym_signal_boost(query: str, hits: list[dict], alpha: float = 0.12) -> list[dict]:
    """
    Generic content-based nudge (no URLs): if a hit mentions expansion terms, add a tiny boost.
    Expects hits like {"title","text","score",...}.
    """
    terms = [t.lower() for t in _terms_for_query(query)]
    if not terms:
        return hits
    for h in hits:
        blob = (h.get("title","") + " " + h.get("text","")).lower()
        if any(term in blob for term in terms):
            h["score"] = h.get("score", 0.0) + alpha
    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits
