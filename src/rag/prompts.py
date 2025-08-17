# TODO: add system/user prompt templates with citation slots (e.g., [^1],[^2])
# src/rag/prompts.py
SYSTEM = """You are a security-savvy documentation assistant. 
Answer concisely based only on the provided context. 
If the answer isn't in context, say you don't have enough information.
Do NOT add a "Sources" section; it will be appended automatically."""

GENERAL_SYSTEM = """You are a helpful assistant.
- For general questions, answer clearly and concisely.
- For security/OWASP topics, be practical and precise.
- Do not fabricate citations. Only include a 'Sources:' section if context is provided."""

USER_TEMPLATE = """Question: {question}

Context:
{context}

Instructions:
- Use only the context.
- Be concise (4-8 sentences).
- If the context doesn't contain the answer, say so.
"""

def build_prompt(question: str, passages: list) -> str:
    ctx_lines = []
    for i, p in enumerate(passages, start=1):
        ctx_lines.append(f"[{i}] {p['title']} — {p['url']}\n{p['text']}\n")
    return USER_TEMPLATE.format(question=question, context="\n".join(ctx_lines))
