from __future__ import annotations
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from config import CONFIG, Config
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _fuzzy_block_ratio(span_n: str, chunk_n: str) -> float:
    """Longest contiguous common block / span length - a partial-ratio
    equivalent that behaves identically under RapidFuzz and difflib."""
    if not span_n:
        return 0.0
    try:
        from rapidfuzz.fuzz import partial_ratio
        return partial_ratio(span_n, chunk_n) / 100.0
    except ImportError:
        m = SequenceMatcher(None, span_n, chunk_n, autojunk=False)
        block = m.find_longest_match(0, len(span_n), 0, len(chunk_n))
        return block.size / len(span_n)


def span_matches_chunk(span: str, chunk_text: str,
                       cfg: Config = CONFIG) -> bool:
    span_n, chunk_n = normalize(span), normalize(chunk_text)
    if not span_n or not chunk_n:
        return False
    if span_n in chunk_n:                                   # (a) containment
        return True
    s_toks = _tokens(span)
    if s_toks:                                              # (b) token recall
        c_set = Counter(_tokens(chunk_text))
        found = sum(min(c, c_set[t]) for t, c in Counter(s_toks).items())
        if found / len(s_toks) >= cfg.span_token_recall:
            return True
    return _fuzzy_block_ratio(span_n, chunk_n) >= cfg.span_fuzzy_ratio  # (c)


def relevant_flags(retrieved_texts: list[str], gold_spans: list[str],
                   cfg: Config = CONFIG) -> list[bool]:
    return [any(span_matches_chunk(s, t, cfg) for s in gold_spans)
            for t in retrieved_texts]

# Retrieval metrics (binary relevance from span matching)

def retrieval_metrics(flags: list[bool], n_gold_spans: int, k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    top = flags[:k]
    hits = sum(top)
    denom = max(min(n_gold_spans, k), 1)
    mrr = 0.0
    for i, f in enumerate(top, 1):
        if f:
            mrr = 1.0 / i
            break
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(top, 1))
    ideal_n = max(hits, 1)          # normalize against the actual hits found
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return {f"precision@{k}": hits / k,
            f"recall@{k}": min(hits, denom) / denom,
            f"hit@{k}": float(hits > 0),
            "mrr": mrr,
            f"ndcg@{k}": min((dcg / idcg) if idcg else 0.0, 1.0)}


def span_coverage(retrieved_texts: list[str], gold_spans: list[str],
                  cfg: Config = CONFIG) -> float:
    if not gold_spans:
        return 0.0
    found = sum(1 for s in gold_spans
                if any(span_matches_chunk(s, t, cfg)
                       for t in retrieved_texts))
    return found / len(gold_spans)

# Similarity metrics (local implementations, dependency-free)
def token_f1(cand: str, ref: str) -> float:
    c, r = Counter(_tokens(cand)), Counter(_tokens(ref))
    ov = sum((c & r).values())
    if not ov:
        return 0.0
    p, rec = ov / max(sum(c.values()), 1), ov / max(sum(r.values()), 1)
    return 2 * p * rec / (p + rec)


def rouge_l(cand: str, ref: str) -> float:
    a, b = _tokens(cand), _tokens(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def bleu(cand: str, ref: str, max_n: int = 4) -> float:
    a, b = _tokens(cand), _tokens(ref)
    if not a or not b:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        an = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
        bn = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
        logs.append(math.log((sum((an & bn).values()) + 1e-9)
                             / max(sum(an.values()), 1)))
    bp = 1.0 if len(a) > len(b) else math.exp(1 - len(b) / max(len(a), 1))
    return bp * math.exp(sum(logs) / max_n)

# Gold set io + v1 migration
REQUIRED_KEYS = {"question", "reference_answer", "gold_spans"}

def load_gold(cfg: Config = CONFIG) -> list[dict]:
    path = Path(cfg.gold_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Author it with gold_tools.py, or migrate the "
            f"v1 set: python gold_tools.py migrate <old_qa_pairs.jsonl>")
    gold = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(f"gold_v2.jsonl line {i} missing {missing}")
        item.setdefault("unanswerable", False)
        if not item["unanswerable"] and not item["gold_spans"]:
            raise ValueError(f"gold_v2.jsonl line {i}: answerable item needs "
                             f">=1 gold span")
        gold.append(item)
    return gold

# Aggregate scoring of one QAResult against one gold item
def score_answer(qa_result, gold: dict, checker=None, embedder=None,
                 cfg: Config = CONFIG, k: int | None = None) -> dict:
    k = k or cfg.top_k
    texts = [(r.parent_text or r.chunk.text) for r in qa_result.retrieved]
    flags = relevant_flags(texts, gold["gold_spans"], cfg) \
        if gold["gold_spans"] else [False] * len(texts)
    row: dict = {"question": gold["question"][:70],
                 "mode": qa_result.mode,
                 "unanswerable": gold["unanswerable"],
                 "refused": qa_result.refused}
    row |= retrieval_metrics(flags, len(gold["gold_spans"]), k)
    row["span_coverage"] = span_coverage(texts, gold["gold_spans"], cfg)

    if gold["unanswerable"]:
        row["refusal_correct"] = float(qa_result.refused)
    else:
        row["refusal_correct"] = float(not qa_result.refused)
        row["rouge_l"] = rouge_l(qa_result.answer, gold["reference_answer"])
        row["bleu"] = bleu(qa_result.answer, gold["reference_answer"])
        row["token_f1"] = token_f1(qa_result.answer, gold["reference_answer"])
        if embedder is not None:
            v = embedder.encode([qa_result.answer, gold["reference_answer"]])
            row["cosine_sim"] = float(v[0] @ v[1])

    if qa_result.faithfulness:
        row["faithfulness"] = qa_result.faithfulness["faithfulness"]
        row["hallucination_rate"] = \
            qa_result.faithfulness["hallucination_rate"]
        row["checker"] = qa_result.faithfulness["checker"]
    elif checker is not None and not qa_result.refused and texts:
        rep = checker.report(qa_result.answer, texts)
        row["faithfulness"] = rep["faithfulness"]
        row["hallucination_rate"] = rep["hallucination_rate"]
        row["checker"] = rep["checker"]

    if qa_result.llm:
        row |= {"latency_s": qa_result.llm.latency_s,
                "tokens_per_s": qa_result.llm.tokens_per_s,
                "model": qa_result.llm.model}
    if qa_result.vram_mb is not None:
        row["vram_mb"] = qa_result.vram_mb
    return row
