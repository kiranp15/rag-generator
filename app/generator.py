"""Grounded answer generation.

Builds a prompt that forces the model to answer only from the retrieved
chunks and to cite which source each claim comes from, then calls the
Anthropic API. Isolated in its own module so the LLM provider/model can be
swapped without touching retrieval or storage code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.config import settings
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a careful research assistant. You answer questions \
using ONLY the numbered source excerpts provided in the user's message.

Rules:
- Base your answer strictly on the provided excerpts. Do not use outside knowledge.
- Every factual claim must be followed by a citation like [1], [2] referring \
to the excerpt number(s) that support it.
- If the excerpts do not contain enough information to answer, say so plainly \
("The provided documents don't contain enough information to answer this.") \
rather than guessing.
- Be concise and directly answer the question first, then add supporting detail.
"""


@dataclass
class GroundedAnswer:
    answer: str
    used_sources: List[str]
    raw_chunks: List[RetrievedChunk]


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c.source}, chunk {c.chunk_index})\n{c.text}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> GroundedAnswer:
    if not chunks:
        return GroundedAnswer(
            answer="I couldn't find anything relevant to this question in the "
            "ingested documents. Try rephrasing, or make sure the right "
            "documents were uploaded to this collection.",
            used_sources=[],
            raw_chunks=[],
        )

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file to enable "
            "answer generation (retrieval works without it)."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context_block = _build_context_block(chunks)
    user_message = (
        f"Source excerpts:\n\n{context_block}\n\n---\n\nQuestion: {question}"
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )

    return GroundedAnswer(
        answer=answer_text,
        used_sources=sorted({c.source for c in chunks}),
        raw_chunks=chunks,
    )
