"""
config.py
=========
Single source of truth for every path and hyperparameter in the project.
Change values HERE, not scattered across scripts.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# PATHS  (everything is relative to the project root, so it works anywhere)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"          # put your extracted "MOOC 1", "MOOC 2"... folders here
PROCESSED_DIR  = DATA_DIR / "processed"    # cleaned text + chunks + FAISS index land here
QA_DIR         = DATA_DIR / "qa"           # question/answer evaluation set
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"  # experiment results (csv/json)

for _d in (RAW_DIR, PROCESSED_DIR, QA_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Intermediate artefact filenames
CHUNKS_FILE      = PROCESSED_DIR / "chunks.jsonl"     # one JSON object per chunk
EMBEDDINGS_FILE  = PROCESSED_DIR / "embeddings.npy"   # float32 matrix [n_chunks, dim]
FAISS_INDEX_FILE = PROCESSED_DIR / "faiss.index"      # serialized FAISS index
BM25_FILE        = PROCESSED_DIR / "bm25.pkl"         # pickled BM25 model + tokens

# ----------------------------------------------------------------------
# CHUNKING
# ----------------------------------------------------------------------
CHUNK_SIZE     = 900     # target characters per chunk
CHUNK_OVERLAP  = 150     # characters shared between consecutive chunks
MIN_CHUNK_CHARS = 80     # drop chunks shorter than this (usually junk)

# ----------------------------------------------------------------------
# EMBEDDING MODEL  (small, fast, strong for retrieval)
# ----------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~80MB
EMBED_BATCH_SIZE = 64

# ----------------------------------------------------------------------
# RETRIEVAL
# ----------------------------------------------------------------------
TOP_K            = 5      # chunks fed to the LLM
HYBRID_ALPHA     = 0.5    # weight of dense vs lexical in hybrid (0=pure BM25, 1=pure dense)
HIER_N_TOPICS    = 2      # hierarchical routing: how many topics to route into (stage 1)

# ----------------------------------------------------------------------
# GENERATOR (SLM)  — pick a small open model. These are gated-free / easy.
#   Good default that needs NO login token:
GEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
#   Alternatives you can swap in (uncomment one):
# GEN_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"   # needs HF login + license accept
# GEN_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"    # ~3.8B, no token needed
# GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"         # even smaller / faster

GEN_MAX_NEW_TOKENS = 400
GEN_TEMPERATURE    = 0.1   # low = more factual / less hallucination
LOAD_IN_4BIT       = True  # auto-disabled if bitsandbytes/GPU unavailable

# ----------------------------------------------------------------------
# BASELINE (unconstrained) — same model, no retrieved context, for comparison
# ----------------------------------------------------------------------
BASELINE_MODEL_NAME = GEN_MODEL_NAME
