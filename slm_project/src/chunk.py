from __future__ import annotations
import re
from config import CONFIG, Config
from store import ChunkRow

HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+\S|\d+(?:\.\d+)*\s+\S|[A-Z][A-Z0-9 \-:]{4,}$)")


def _is_heading(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) < 90 and bool(HEADING_RE.match(s))


def chunk_one_page(rec: dict, cfg: Config = CONFIG) -> list[ChunkRow]:
    """Split one parent (page/slide) into overlapping child chunks; a heading
    always starts a new child so children never straddle sections."""
    size, ov = cfg.child_chunk_tokens, cfg.child_overlap_tokens
    assert 0 <= ov < size, "overlap must be smaller than chunk size"
    parent_id = f"{rec['source']}:p{rec['page']}"

    children: list[ChunkRow] = []
    buf: list[str] = []
    heading = ""

    def flush(carry_overlap: bool) -> None:
        nonlocal buf
        if not buf:
            return
        cid = f"{parent_id}:c{len(children):02d}"
        children.append(ChunkRow(
            chunk_id=cid, parent_id=parent_id, source=rec["source"],
            page=rec["page"], text=" ".join(buf), context=None,
            n_tokens=len(buf)))
        buf = buf[-ov:] if (carry_overlap and ov) else []

    for line in rec["text"].splitlines():
        if _is_heading(line):
            flush(carry_overlap=False)      # hard boundary: no cross-section overlap
            heading = line.strip().lstrip("# ")
            buf = heading.split()           # heading text leads its section
            continue
        toks, i = line.split(), 0
        while i < len(toks):
            take = toks[i:i + size - len(buf)]
            buf.extend(take)
            i += len(take)
            if len(buf) >= size:
                flush(carry_overlap=True)
    flush(carry_overlap=False)
    return children


def build_parent_and_children(pages: list[dict],
                              cfg: Config = CONFIG
                              ) -> tuple[list[dict], list[ChunkRow]]:
    """From per-page records -> (parents for the store, child ChunkRows)."""
    parents, children = [], []
    for rec in pages:
        first = next((l.strip().lstrip("# ") for l in rec["text"].splitlines()
                      if _is_heading(l)), "")
        parents.append({"parent_id": f"{rec['source']}:p{rec['page']}",
                        "source": rec["source"], "page": rec["page"],
                        "heading": first, "text": rec["text"]})
        children.extend(chunk_one_page(rec, cfg))
    return parents, children
