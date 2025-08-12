# TODO: add system/user prompt templates with citation slots (e.g., [^1],[^2])
# src/rag/prompts.py
SYSTEM = """You are a security-savvy documentation assistant. 
Answer concisely based only on the provided context. 
If the answer isn't in context, say you don't have enough information."""
USER_TEMPLATE = """Question: {question}

Context:
{context}

Instructions:
- Use only the context.
- Be concise (4-8 sentences).
- After the answer, add a "Sources:" list with [n] and the URLs.
"""

def build_prompt(question: str, passages: list) -> str:
    ctx_lines = []
    for i, p in enumerate(passages, start=1):
        ctx_lines.append(f"[{i}] {p['title']} — {p['url']}\n{p['text']}\n")
    return USER_TEMPLATE.format(question=question, context="\n".join(ctx_lines))
