"""store.py — the single SQLite database behind the whole pipeline."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import CONFIG, Config

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS files (
    path      TEXT PRIMARY KEY,
    sha256    TEXT NOT NULL,
    n_parents INTEGER NOT NULL DEFAULT 0,
    n_chunks  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,          -- "file.pdf:p12"
    source    TEXT NOT NULL,
    page      INTEGER NOT NULL,
    heading   TEXT NOT NULL DEFAULT '',
    text      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,          -- "file.pdf:p12:c03"
    parent_id TEXT NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
    source    TEXT NOT NULL,
    page      INTEGER NOT NULL,
    text      TEXT NOT NULL,
    context   TEXT,                      -- LLM situating context (contextual retrieval)
    n_tokens  INTEGER NOT NULL,
    embedding BLOB                       -- float16 vector, NULL until embedded
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, body, tokenize='porter unicode61'
);
"""


@dataclass
class ChunkRow:
    chunk_id: str
    parent_id: str
    source: str
    page: int
    text: str
    context: str | None
    n_tokens: int

    @property
    def indexed_text(self) -> str:
        """Text used for BOTH embedding and BM25: situating context + chunk.
        This is Anthropic-style contextual retrieval when context is set."""
        return f"{self.context}\n{self.text}" if self.context else self.text

    @property
    def citation(self) -> str:
        return f"[{self.source}, p.{self.page}]"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Store:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = sqlite3.connect(cfg.db_path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)

    def file_changed(self, path: Path) -> bool:
        row = self.db.execute("SELECT sha256 FROM files WHERE path=?",
                              (path.name,)).fetchone()
        return row is None or row[0] != file_sha256(path)

    def stale_files(self, present: set[str]) -> list[str]:
        """Files in the DB no longer present on disk (deleted from corpus)."""
        rows = self.db.execute("SELECT path FROM files").fetchall()
        return [r[0] for r in rows if r[0] not in present]

    def remove_file(self, filename: str) -> None:
        with self.db:
            self.db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE source=?)", (filename,))
            self.db.execute("DELETE FROM chunks  WHERE source=?", (filename,))
            self.db.execute("DELETE FROM parents WHERE source=?", (filename,))
            self.db.execute("DELETE FROM files   WHERE path=?", (filename,))

    def replace_file(self, path: Path, parents: list[dict],
                     chunks: list[ChunkRow]) -> None:
        """Atomic re-ingestion of one file (incremental indexing unit)."""
        with self.db:
            self.remove_file(path.name)
            self.db.executemany(
                "INSERT INTO parents(parent_id,source,page,heading,text) "
                "VALUES(:parent_id,:source,:page,:heading,:text)", parents)
            self.db.executemany(
                "INSERT INTO chunks(chunk_id,parent_id,source,page,text,"
                "context,n_tokens) VALUES(?,?,?,?,?,?,?)",
                [(c.chunk_id, c.parent_id, c.source, c.page, c.text,
                  c.context, c.n_tokens) for c in chunks])
            self.db.executemany(
                "INSERT INTO chunks_fts(chunk_id,body) VALUES(?,?)",
                [(c.chunk_id, c.indexed_text) for c in chunks])
            self.db.execute(
                "INSERT INTO files(path,sha256,n_parents,n_chunks) "
                "VALUES(?,?,?,?)",
                (path.name, file_sha256(path), len(parents), len(chunks)))

    def n_chunks(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunk(self, chunk_id: str) -> ChunkRow | None:
        r = self.db.execute(
            "SELECT chunk_id,parent_id,source,page,text,context,n_tokens "
            "FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return ChunkRow(*r) if r else None

    def get_parent_text(self, parent_id: str, max_tokens: int) -> str:
        r = self.db.execute("SELECT text FROM parents WHERE parent_id=?",
                            (parent_id,)).fetchone()
        if not r:
            return ""
        words = r[0].split()
        return " ".join(words[:max_tokens])

    def iter_chunks(self, only_unembedded: bool = False):
        q = ("SELECT chunk_id,parent_id,source,page,text,context,n_tokens "
             "FROM chunks" + (" WHERE embedding IS NULL" if only_unembedded
                              else "") + " ORDER BY chunk_id")
        for r in self.db.execute(q):
            yield ChunkRow(*r)

    def sample_chunks(self, n: int, min_tokens: int = 60) -> list[ChunkRow]:
        rows = self.db.execute(
            "SELECT chunk_id,parent_id,source,page,text,context,n_tokens "
            "FROM chunks WHERE n_tokens>=? ORDER BY RANDOM() LIMIT ?",
            (min_tokens, n)).fetchall()
        return [ChunkRow(*r) for r in rows]

    def save_embeddings(self, ids: list[str], vecs: np.ndarray) -> None:
        assert len(ids) == len(vecs)
        with self.db:
            self.db.executemany(
                "UPDATE chunks SET embedding=? WHERE chunk_id=?",
                [(v.astype(np.float16).tobytes(), cid)
                 for cid, v in zip(ids, vecs)])

    def load_embedding_matrix(self) -> tuple[list[str], np.ndarray]:
        """All cached vectors, ordered — the HNSW index rebuilds from this in
        seconds, so incremental corpus changes never force re-embedding."""
        rows = self.db.execute(
            "SELECT chunk_id, embedding FROM chunks "
            "WHERE embedding IS NOT NULL ORDER BY chunk_id").fetchall()
        if not rows:
            return [], np.zeros((0, self.cfg.embed_dim), dtype=np.float32)
        ids = [r[0] for r in rows]
        mat = np.stack([np.frombuffer(r[1], dtype=np.float16).astype(np.float32)
                        for r in rows])
        return ids, mat

    @staticmethod
    def _fts_query(query: str, max_terms: int = 30) -> str:
        """Sanitise arbitrary text into a safe FTS5 OR-query."""
        import re
        terms = re.findall(r"[A-Za-z0-9]+", query)[:max_terms]
        return " OR ".join(f'"{t}"' for t in terms) if terms else '""'

    def bm25_search(self, query: str, top_n: int) -> list[tuple[str, int]]:
        """Returns [(chunk_id, rank)] — FTS5 bm25() is smaller-is-better, and
        RRF fusion only needs ranks, so we never touch raw score semantics."""
        q = self._fts_query(query)
        rows = self.db.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?", (q, top_n)).fetchall()
        return [(r[0], rank) for rank, r in enumerate(rows)]

    def chunks_missing_context(self, limit: int | None = None) -> list[ChunkRow]:
        q = ("SELECT chunk_id,parent_id,source,page,text,context,n_tokens "
             "FROM chunks WHERE context IS NULL ORDER BY chunk_id")
        if limit:
            q += f" LIMIT {int(limit)}"
        return [ChunkRow(*r) for r in self.db.execute(q)]

    def set_context(self, chunk_id: str, context: str) -> None:
        """Store situating context and refresh the FTS row to include it."""
        with self.db:
            self.db.execute("UPDATE chunks SET context=?, embedding=NULL "
                            "WHERE chunk_id=?", (context, chunk_id))
            self.db.execute("DELETE FROM chunks_fts WHERE chunk_id=?",
                            (chunk_id,))
            row = self.get_chunk(chunk_id)
            self.db.execute("INSERT INTO chunks_fts(chunk_id,body) VALUES(?,?)",
                            (chunk_id, row.indexed_text))

    def doc_head(self, source: str, max_chars: int) -> str:
        rows = self.db.execute(
            "SELECT text FROM parents WHERE source=? ORDER BY page LIMIT 5",
            (source,)).fetchall()
        return "\n".join(r[0] for r in rows)[:max_chars]

    def close(self) -> None:
        self.db.close()
