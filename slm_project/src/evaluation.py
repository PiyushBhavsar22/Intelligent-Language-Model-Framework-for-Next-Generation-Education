"""
evaluation.py
=============
All metrics named in the practicum proposal:

Retrieval    : precision@k, recall@k        (needs gold source per question)
Grounding    : fraction of answer sentences supported by the retrieved context
Text overlap : ROUGE-L, BLEU, token-F1, embedding cosine similarity vs reference
Performance  : timed at call sites (see run scripts)

Everything is dependency-light and CPU-friendly.
"""
from __future__ import annotations
import re
from functools import lru_cache

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.config import EMBED_MODEL_NAME

# ensure the tiny tokenizer data BLEU needs is present (safe to call repeatedly)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ----------------------------------------------------------------------
# shared embedder (reused so we don't reload the model per call)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


# ----------------------------------------------------------------------
# RETRIEVAL METRICS
# ----------------------------------------------------------------------
def precision_recall_at_k(retrieved_sources: list[str],
                          gold_sources: set[str], k: int) -> tuple[float, float]:
    """
    retrieved_sources : ordered list of chunk 'source' strings from the retriever
    gold_sources      : set of source filenames that DO contain the answer
    Returns (precision@k, recall@k).
    """
    topk = retrieved_sources[:k]
    if not topk:
        return 0.0, 0.0
    hits = sum(1 for s in topk if s in gold_sources)
    precision = hits / len(topk)
    recall = hits / len(gold_sources) if gold_sources else 0.0
    return precision, recall


# ----------------------------------------------------------------------
# GROUNDING
# ----------------------------------------------------------------------
def grounding_score(answer: str, contexts: list[dict],
                    threshold: float = 0.55) -> float:
    """
    Fraction of answer sentences whose max cosine similarity to any context
    chunk exceeds `threshold`. 1.0 = fully grounded, 0.0 = fully hallucinated.
    A high grounding score == low hallucination.
    """
    ans_sents = _sentences(answer)
    if not ans_sents:
        return 0.0
    ctx_texts = [c["text"] for c in contexts] or [""]
    emb = _embedder()
    a = emb.encode(ans_sents, convert_to_numpy=True, normalize_embeddings=True)
    c = emb.encode(ctx_texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(a, c)          # [n_ans, n_ctx]
    max_per_sent = sims.max(axis=1)
    return float((max_per_sent >= threshold).mean())


def hallucination_rate(answer: str, contexts: list[dict],
                       threshold: float = 0.55) -> float:
    """Convenience inverse of grounding: 1 - grounding_score."""
    return 1.0 - grounding_score(answer, contexts, threshold)


# ----------------------------------------------------------------------
# TEXT-SIMILARITY vs REFERENCE ANSWER
# ----------------------------------------------------------------------
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_SMOOTH = SmoothingFunction().method1


def rouge_l(pred: str, ref: str) -> float:
    return _ROUGE.score(ref, pred)["rougeL"].fmeasure


def bleu(pred: str, ref: str) -> float:
    ref_toks, pred_toks = _tokens(ref), _tokens(pred)
    if not pred_toks or not ref_toks:
        return 0.0
    return float(sentence_bleu([ref_toks], pred_toks, smoothing_function=_SMOOTH))


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    common = {}
    for t in p:
        if t in r:
            common[t] = common.get(t, 0) + 1
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(r)
    return 2 * precision * recall / (precision + recall)


def answer_cosine(pred: str, ref: str) -> float:
    emb = _embedder()
    v = emb.encode([pred, ref], convert_to_numpy=True, normalize_embeddings=True)
    return float(cosine_similarity(v[0:1], v[1:2])[0, 0])


def text_similarity_bundle(pred: str, ref: str) -> dict:
    return {
        "rougeL": round(rouge_l(pred, ref), 4),
        "bleu": round(bleu(pred, ref), 4),
        "token_f1": round(token_f1(pred, ref), 4),
        "cosine": round(answer_cosine(pred, ref), 4),
    }
