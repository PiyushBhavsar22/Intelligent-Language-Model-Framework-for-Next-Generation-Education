from __future__ import annotations

import logging
import re
import warnings

import numpy as np

from config import CONFIG, Config

log = logging.getLogger("faithfulness")

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[[^\]]+\]")


def split_sentences(text: str) -> list[str]:
    sents = [_CITATION.sub("", s).strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in sents
            if len(re.findall(r"[a-z0-9]+", s.lower())) >= 3]


class FaithfulnessChecker:
    _HHEM_PREMISE_TOKEN_BUDGET = 400
    _HHEM_HYPOTHESIS_TOKEN_BUDGET = 80
    _HHEM_WINDOW_OVERLAP_TOKENS = 80

    def __init__(self, cfg: Config = CONFIG, embedder=None):
        self.cfg = cfg
        self.embedder = embedder      # v1-style fallback path
        self._hhem = None
        self._hhem_failed = False

    # backends
    def _load_hhem(self):
        if self._hhem is None and not self._hhem_failed:
            try:
                from transformers import AutoModelForSequenceClassification
                self._hhem = AutoModelForSequenceClassification.from_pretrained(
                    self.cfg.hhem_model_name, trust_remote_code=True)
                self._hhem.eval()
                log.info("HHEM-2.1-Open loaded (CPU faithfulness gate).")
            except Exception as exc:
                self._hhem_failed = True
                log.warning("HHEM unavailable (%s) - falling back to "
                            "embedding-similarity grounding.", exc)
        return self._hhem

    def _hhem_tokenizer(self):
        model = self._load_hhem()
        return getattr(model, "tokenizer", None) if model else None

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tok = self._hhem_tokenizer()
        if tok is None:
            return text[: max_tokens * 4]      # conservative char fallback
        ids = tok.encode(text, add_special_tokens=False,
                         truncation=True, max_length=max_tokens)
        return tok.decode(ids, skip_special_tokens=True)

    def _premise_windows(self, block: str) -> list[str]:
        tok = self._hhem_tokenizer()
        if tok is None:
            # char-based fallback windows (~4 chars/token)
            size = self._HHEM_PREMISE_TOKEN_BUDGET * 4
            step = size - self._HHEM_WINDOW_OVERLAP_TOKENS * 4
            return [block[i:i + size] for i in range(0, max(len(block), 1), step)]
        ids = tok.encode(block, add_special_tokens=False)
        if len(ids) <= self._HHEM_PREMISE_TOKEN_BUDGET:
            return [block]
        step = self._HHEM_PREMISE_TOKEN_BUDGET - self._HHEM_WINDOW_OVERLAP_TOKENS
        windows = []
        for start in range(0, len(ids), step):
            piece = ids[start:start + self._HHEM_PREMISE_TOKEN_BUDGET]
            if not piece:
                break
            windows.append(tok.decode(piece, skip_special_tokens=True))
            if start + self._HHEM_PREMISE_TOKEN_BUDGET >= len(ids):
                break
        return windows

    def _hhem_scores(self, premise: str, sentences: list[str]) -> np.ndarray:
        model = self._load_hhem()
        pairs = [(premise, s) for s in sentences]
        scores = model.predict(pairs)          # HHEM exposes .predict
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def _embed_scores(self, premise_chunks: list[str],
                      sentences: list[str]) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("No embedder provided for fallback grounding.")
        sv = self.embedder.encode(sentences)
        cv = self.embedder.encode(premise_chunks)
        return (sv @ cv.T).max(axis=1)

    # public
    def report(self, answer: str, context_blocks: list[str]) -> dict:
        """{'faithfulness', 'hallucination_rate', 'checker', 'unsupported'}"""
        sents = split_sentences(answer)
        if not sents:
            return {"faithfulness": 1.0, "hallucination_rate": 0.0,
                    "checker": "none", "n_sentences": 0, "unsupported": []}

        if self._load_hhem() is not None:
            sents_trunc = [self._truncate_to_tokens(
                s, self._HHEM_HYPOTHESIS_TOKEN_BUDGET) for s in sents]
            per_window: list[np.ndarray] = []
            for block in context_blocks:
                for window in self._premise_windows(block):
                    per_window.append(self._hhem_scores(window, sents_trunc))
            if per_window:
                scores = np.max(np.stack(per_window), axis=0)
            else:
                scores = np.zeros(len(sents), dtype=np.float32)
            checker = "hhem"
        else:
            scores, checker = self._embed_scores(context_blocks, sents), "embed"

        thr = self.cfg.faithfulness_threshold
        unsupported = [s for s, sc in zip(sents, scores) if sc < thr]
        rate = len(unsupported) / len(sents)
        return {"faithfulness": float(np.mean(scores)),
                "hallucination_rate": rate, "checker": checker,
                "n_sentences": len(sents), "unsupported": unsupported}