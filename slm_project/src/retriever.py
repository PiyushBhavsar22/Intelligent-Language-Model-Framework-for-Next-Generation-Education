"""
retriever.py
============
Four retrieval strategies over the module corpus:
  * dense        : sentence-transformer embeddings + FAISS cosine search
  * lexical      : BM25 keyword search
  * hybrid       : score-normalised blend of the two (weighted by HYBRID_ALPHA)
  * hierarchical : two-stage document routing (RQ4). Stage 1 routes the query
                   to the most relevant topic folders using topic-level
                   centroid embeddings; stage 2 runs hybrid search restricted
                   to chunks inside those topics. Reduces retrieval noise by
                   excluding off-topic material before chunk-level ranking.

Build the index once (build_index), then load + query it repeatedly.
"""
from __future__ import annotations
import json
import pickle
import re
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE, BM25_FILE,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, TOP_K, HYBRID_ALPHA, HIER_N_TOPICS,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ----------------------------------------------------------------------
# BUILD
# ----------------------------------------------------------------------
def build_index(chunks: list[dict]) -> None:
    """Embed all chunks, build + persist FAISS and BM25 indexes."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,      # so inner product == cosine similarity
    ).astype("float32")

    # ---- FAISS (inner product on normalised vectors = cosine) ----
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    np.save(EMBEDDINGS_FILE, emb)

    # ---- BM25 ----
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25}, f)

    # ---- persist chunks ----
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Index built: {len(chunks)} chunks, dim={emb.shape[1]}")


# ----------------------------------------------------------------------
# LOAD + QUERY
# ----------------------------------------------------------------------
class Retriever:
    def __init__(self) -> None:
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                "No index found. Run scripts/1_build_index.py first."
            )
        self.chunks = [json.loads(l) for l in open(CHUNKS_FILE, encoding="utf-8")]
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        # ---- hierarchical routing structures (built from saved embeddings) ----
        # topic key = "module/topic" so identical topic names in different
        # modules stay distinct. Centroid = mean of that topic's chunk vectors.
        self._emb = np.load(EMBEDDINGS_FILE)  # [n_chunks, dim], L2-normalised
        self._topic_of = np.array(
            [f"{c['module']}/{c['topic']}" for c in self.chunks]
        )
        self._topics = sorted(set(self._topic_of))
        cents = []
        for t in self._topics:
            v = self._emb[self._topic_of == t].mean(axis=0)
            n = np.linalg.norm(v)
            cents.append(v / n if n > 0 else v)
        self._topic_centroids = np.vstack(cents).astype("float32")

    # ---- individual strategies ----
    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, len(self.chunks))
        out = np.zeros(len(self.chunks), dtype="float32")
        out[idxs[0]] = scores[0]
        return out

    def _lexical_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(_tokenize(query)), dtype="float32")

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    # ---- hierarchical routing (stage 1) ----
    def route_topics(self, query: str, n_topics: int = HIER_N_TOPICS) -> list[str]:
        """Return the n_topics topic keys whose centroids best match the query."""
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")[0]
        sims = self._topic_centroids @ q
        order = np.argsort(sims)[::-1][:n_topics]
        return [self._topics[int(i)] for i in order]

    # ---- public API ----
    def search(self, query: str, top_k: int = TOP_K,
               method: str = "hybrid", alpha: float = HYBRID_ALPHA) -> list[dict]:
        """
        method: 'dense' | 'lexical' | 'hybrid' | 'hierarchical'
        Returns top_k chunk dicts, each with an added 'score' field.
        """
        if method == "dense":
            scores = self._dense_scores(query)
        elif method == "lexical":
            scores = self._lexical_scores(query)
        elif method in ("hybrid", "hierarchical"):
            d = self._minmax(self._dense_scores(query))
            l = self._minmax(self._lexical_scores(query))
            scores = alpha * d + (1 - alpha) * l
            if method == "hierarchical":
                # stage 2: mask out every chunk outside the routed topics
                allowed = set(self.route_topics(query))
                mask = np.isin(self._topic_of, list(allowed))
                scores = np.where(mask, scores, -1.0)
        else:
            raise ValueError(f"unknown method: {method}")

        top = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top:
            c = dict(self.chunks[int(i)])
            c["score"] = float(scores[int(i)])
            results.append(c)
        return results
