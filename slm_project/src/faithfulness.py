from __future__ import annotations
import logging
import re
import numpy as np
from config import CONFIG, Config

log = logging.getLogger("faithfulness")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[[^\]]+\]")


def split_sentences(text: str) -> list[str]:
    sents = [_CITATION.sub("", s).strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in sents
            if len(re.findall(r"[a-z0-9]+", s.lower())) >= 3]

class FaithfulnessChecker:
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
        premise = "\n".join(context_blocks)[:1800]
        if self._load_hhem() is not None:
            scores, checker = self._hhem_scores(premise, sents), "hhem"
        else:
            scores, checker = self._embed_scores(context_blocks, sents), "embed"
        thr = self.cfg.faithfulness_threshold
        unsupported = [s for s, sc in zip(sents, scores) if sc < thr]
        rate = len(unsupported) / len(sents)
        return {"faithfulness": float(np.mean(scores)),
                "hallucination_rate": rate, "checker": checker,
                "n_sentences": len(sents), "unsupported": unsupported}
