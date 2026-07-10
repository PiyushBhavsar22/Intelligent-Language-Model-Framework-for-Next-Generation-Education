"""
chunk.py
========
Split cleaned documents into overlapping character-based chunks.

Why character-based (not token-based)?
  * Simple, dependency-free, deterministic, and fast.
  * Overlap preserves context across boundaries so answers aren't cut in half.
We split on paragraph/sentence boundaries where possible to avoid mid-word cuts.
"""
from __future__ import annotations
import re
from typing import Iterator

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence/paragraph splitter (no heavy NLP dependency)."""
    # First break on blank lines, then on sentence enders.
    blocks = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # split on . ? ! followed by space+capital, keep the delimiter
        sents = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", b)
        pieces.extend(s.strip() for s in sents if s.strip())
    return pieces


def chunk_document(doc: dict) -> Iterator[dict]:
    """
    Yield chunk dicts for a single document:
        {chunk_id, source, module, topic, text}
    Greedily packs sentences up to CHUNK_SIZE, then starts a new chunk that
    overlaps the tail of the previous one by ~CHUNK_OVERLAP characters.
    """
    sentences = _split_sentences(doc["text"])
    if not sentences:
        return

    buf = ""
    idx = 0
    for sent in sentences:
        if buf and len(buf) + 1 + len(sent) > CHUNK_SIZE:
            # flush current chunk
            if len(buf) >= MIN_CHUNK_CHARS:
                yield _make_chunk(doc, idx, buf)
                idx += 1
            # start next chunk with overlapping tail
            tail = buf[-CHUNK_OVERLAP:]
            # don't cut a word in half at the overlap boundary
            sp = tail.find(" ")
            if sp != -1:
                tail = tail[sp + 1:]
            buf = (tail + " " + sent).strip()
        else:
            buf = (buf + " " + sent).strip() if buf else sent

    if len(buf) >= MIN_CHUNK_CHARS:
        yield _make_chunk(doc, idx, buf)


def _make_chunk(doc: dict, idx: int, text: str) -> dict:
    return {
        "chunk_id": f"{doc['source']}::{idx}",
        "source": doc["source"],
        "module": doc["module"],
        "topic": doc["topic"],
        "text": text.strip(),
    }
