"""retriever.py — the two-stage retrieval core of the v2 upgrade.

Stage 1  hybrid recall:  dense HNSW top-50  +  FTS5-BM25 top-50, fused with
         weighted Reciprocal Rank Fusion (rank-based, so BM25 score semantics
         never matter).
Stage 2  precision:      bge-reranker-v2-m3 cross-encoder scores (query,
         chunk) jointly over the fused candidates; keep top-k (5)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import CONFIG, Config
from embed_index import Embedder, HnswIndex
from store import Store, ChunkRow

log = logging.getLogger("retriever")


@dataclass
class Retrieved:
    chunk: ChunkRow
    fused_score: float          # RRF score (stage 1)
    rerank_score: float | None  # cross-encoder prob (stage 2); None if off
    parent_text: str = ""       # filled by parent expansion

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def best_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None \
            else self.fused_score


class Reranker:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.cfg.reranker_model_name,
                                       max_length=self.cfg.rerank_max_length,
                                       device=self.cfg.reranker_device)
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        raw = self._load().predict([(query, t) for t in texts],
                                   convert_to_numpy=True)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.size and (raw.min() < 0.0 or raw.max() > 1.0):  # logits -> prob
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw

class Retriever:
    def __init__(self, store: Store, index: HnswIndex, embedder: Embedder,
                 reranker: Reranker | None = None, cfg: Config = CONFIG):
        self.store, self.index, self.embedder = store, index, embedder
        self.reranker = reranker or Reranker(cfg)
        self.cfg = cfg

    # stage 1
    def _hybrid_candidates(self, query: str, mode: str) -> dict[str, float]:
        cfg = self.cfg
        dense_hits: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid"):
            qv = self.embedder.encode([query])
            dense_hits = self.index.search(qv, cfg.dense_top_n)[0]
        lex_hits: list[tuple[str, int]] = []
        if mode in ("lexical", "hybrid"):
            lex_hits = self.store.bm25_search(query, cfg.lexical_top_n)

        fused: dict[str, float] = {}

        def add_rrf(ranked_ids: list[str], weight: float) -> None:
            for rank, cid in enumerate(ranked_ids):
                fused[cid] = fused.get(cid, 0.0) + weight / (cfg.rrf_k + rank + 1)

        if mode == "dense":
            add_rrf([cid for cid, _ in dense_hits], 1.0)
        elif mode == "lexical":
            add_rrf([cid for cid, _ in lex_hits], 1.0)
        else:
            add_rrf([cid for cid, _ in dense_hits], cfg.dense_weight)
            add_rrf([cid for cid, _ in lex_hits], cfg.lexical_weight)

        # cheap early-exit floor
        if mode != "lexical" and dense_hits:
            if dense_hits[0][1] < cfg.dense_refusal_floor and not lex_hits:
                return {}
        return fused

    # public
    def retrieve(self, query: str, top_k: int | None = None,
                 mode: str = "hybrid", use_reranker: bool = True,
                 expand_parents: bool = True,
                 refusal_threshold: float | None = None) -> list[Retrieved]:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        thr = (cfg.rerank_refusal_threshold if refusal_threshold is None
               else refusal_threshold)

        fused = self._hybrid_candidates(query, mode)
        if not fused:
            return []
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        cand_n = cfg.rerank_candidates if use_reranker else top_k
        cand_ids = [cid for cid, _ in ranked[:cand_n]]
        rows = [self.store.get_chunk(cid) for cid in cand_ids]
        rows = [(cid, r) for cid, r in zip(cand_ids, rows) if r is not None]

        results: list[Retrieved]
        if use_reranker:
            scores = self.reranker.score(query,
                                         [r.indexed_text for _, r in rows])
            order = np.argsort(scores)[::-1][:top_k]
            results = [Retrieved(rows[i][1], fused[rows[i][0]],
                                 float(scores[i])) for i in order]
            if not results or results[0].rerank_score < thr:
                return []                              # out-of-scope gate
        else:
            results = [Retrieved(r, fused[cid], None)
                       for cid, r in rows[:top_k]]

        if expand_parents:
            for res in results:
                res.parent_text = self.store.get_parent_text(
                    res.chunk.parent_id, cfg.parent_max_tokens)
        return results


def build_context(results: list[Retrieved], max_chars: int = 12000) -> str:
    seen: set[str] = set()
    parts, used = [], 0
    for r in results:
        key = r.chunk.parent_id if r.parent_text else r.chunk_id
        if key in seen:
            continue
        seen.add(key)
        body = r.parent_text or r.chunk.text
        block = (f"--- SOURCE {r.chunk.citation} (id={key}) ---\n{body}")
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
