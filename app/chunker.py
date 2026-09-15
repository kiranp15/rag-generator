"""Splits document text into overlapping, retrieval-sized chunks.

Strategy: split on paragraph boundaries first (so we don't cut mid-sentence
whenever possible), then greedily pack paragraphs into windows of roughly
`chunk_size` characters, carrying `chunk_overlap` characters of trailing
context into the next chunk so answers near a chunk boundary aren't lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    index: int
    source: str


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Chunk]:
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph longer than chunk_size gets hard-split.
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), chunk_size - chunk_overlap):
                chunks.append(para[i : i + chunk_size])
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # carry overlap from the tail of the previous chunk
            overlap_text = current[-chunk_overlap:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip() if overlap_text else para

    if current:
        chunks.append(current)

    return [
        Chunk(text=c.strip(), index=i, source=source)
        for i, c in enumerate(chunks)
        if c.strip()
    ]
