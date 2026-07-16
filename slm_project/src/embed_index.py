from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from config import CONFIG, Config
from store import Store

log = logging.getLogger("embed_index")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Embedder:
    def __init__(self, cfg: Config = CONFIG):
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embed_model_name,
                                         device=cfg.embed_device)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != cfg.embed_dim:
            raise ValueError(
                f"Config embed_dim={cfg.embed_dim} but {cfg.embed_model_name} "
                f"produces {dim}-dim vectors - fix config.embed_dim.")

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.embed_dim), dtype=np.float32)
        v = self.model.encode(texts, batch_size=self.cfg.embed_batch_size,
                              normalize_embeddings=True,
                              show_progress_bar=show_progress)
        return np.asarray(v, dtype=np.float32)


class HnswIndex:

    def __init__(self, index, ids: list[str], cfg: Config):
        self.index, self.ids, self.cfg = index, ids, cfg

    @classmethod
    def build(cls, ids: list[str], mat: np.ndarray,
              cfg: Config = CONFIG) -> "HnswIndex":
        import faiss
        if len(ids) == 0:
            raise ValueError("Cannot build an index over an empty corpus - "
                             "run ingest.py and embed_index.py first.")
        index = faiss.IndexHNSWFlat(mat.shape[1], cfg.hnsw_m,
                                    faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.hnsw_ef_construction
        index.hnsw.efSearch = cfg.hnsw_ef_search
        index.add(mat)
        obj = cls(index, ids, cfg)
        obj.save()
        return obj

    def save(self) -> None:
        import faiss
        self.cfg.ensure_dirs()
        faiss.write_index(self.index, str(self.cfg.hnsw_path))
        ids_path = Path(str(self.cfg.hnsw_path) + ".ids")
        ids_path.write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, cfg: Config = CONFIG) -> "HnswIndex":
        import faiss
        ids_path = Path(str(cfg.hnsw_path) + ".ids")
        if not cfg.hnsw_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"No HNSW index at {cfg.hnsw_path} - build it: "
                f"python ingest.py && python embed_index.py")
        index = faiss.read_index(str(cfg.hnsw_path))
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        if index.ntotal != len(ids):
            raise RuntimeError("Index/ids mismatch - rerun embed_index.py")
        index.hnsw.efSearch = cfg.hnsw_ef_search
        return cls(index, ids, cfg)

    def search(self, qvecs: np.ndarray, top_n: int
               ) -> list[list[tuple[str, float]]]:
        if qvecs.ndim == 1:
            qvecs = qvecs[None, :]
        if qvecs.shape[1] != self.index.d:
            raise ValueError(f"Query dim {qvecs.shape[1]} != index dim "
                             f"{self.index.d} - embedding model changed? "
                             f"Rebuild: python embed_index.py")
        D, I = self.index.search(qvecs.astype(np.float32),
                                 min(top_n, len(self.ids)))
        return [[(self.ids[i], float(d)) for d, i in zip(drow, irow) if i >= 0]
                for drow, irow in zip(D, I)]


def build_or_update(cfg: Config = CONFIG) -> HnswIndex:
    """Embed only what's missing, then rebuild HNSW from the cached matrix."""
    store = Store(cfg)
    pending = list(store.iter_chunks(only_unembedded=True))
    if pending:
        log.info("Embedding %d new/updated chunks with %s on %s ...",
                 len(pending), cfg.embed_model_name, cfg.embed_device)
        emb = Embedder(cfg)
        B = 512
        for i in range(0, len(pending), B):
            batch = pending[i:i + B]
            vecs = emb.encode([c.indexed_text for c in batch],
                              show_progress=True)
            store.save_embeddings([c.chunk_id for c in batch], vecs)
            log.info("  cached %d/%d", min(i + B, len(pending)), len(pending))
    ids, mat = store.load_embedding_matrix()
    log.info("Rebuilding HNSW over %d vectors (seconds, from cache)...",
             len(ids))
    index = HnswIndex.build(ids, mat, cfg)
    store.close()
    log.info("Index ready: %d vectors, dim %d.", index.index.ntotal,
             index.index.d)
    return index


if __name__ == "__main__":
    build_or_update()
