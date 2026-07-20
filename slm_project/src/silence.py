from __future__ import annotations
import logging
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

for _name in (
    "faiss", "faiss.loader",
    "sentence_transformers", "sentence_transformers.SentenceTransformer",
    "sentence_transformers.cross_encoder.CrossEncoder",
    "transformers", "transformers.tokenization_utils_base",
    "transformers.modeling_utils",
    "urllib3", "httpx",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

try:
    from functools import partialmethod

    from tqdm import tqdm as _tqdm
    _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
except Exception:
    pass