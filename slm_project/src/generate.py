from __future__ import annotations

import silence 

import argparse
import logging
from dataclasses import dataclass, field

from config import CONFIG, Config
from faithfulness import FaithfulnessChecker
from llm import OllamaClient, LLMResult, gpu_memory_mb
from prompt import build_prompt, is_refusal
from retriever import Retriever, Retrieved, build_context

log = logging.getLogger("generate")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

GROUNDED_TEMPERATURE = 0.3   


@dataclass
class QAResult:
    question: str
    answer: str
    mode: str
    refused: bool
    retrieved: list[Retrieved] = field(default_factory=list)
    llm: LLMResult | None = None
    faithfulness: dict | None = None
    regenerated: bool = False
    flagged: bool = False         
    vram_mb: float | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [r.chunk_id for r in self.retrieved]


class Pipeline:
    def __init__(self, retriever: Retriever | None, client: OllamaClient,
                 checker: FaithfulnessChecker | None = None,
                 cfg: Config = CONFIG):
        self.retriever, self.client, self.cfg = retriever, client, cfg
        self.checker = checker

    def answer(self, question: str, mode: str = "strict",
               use_reranker: bool = True, expand_parents: bool = True,
               use_gate: bool = True, top_k: int | None = None,
               refusal_threshold: float | None = None) -> QAResult:
        cfg = self.cfg

        if mode == "baseline":
            sys, usr = build_prompt(question, "", "baseline", cfg=cfg)
            res = self.client.generate(sys, usr,
                                       temperature=GROUNDED_TEMPERATURE)
            return QAResult(question, res.text, mode,
                            is_refusal(res.text, cfg), llm=res,
                            vram_mb=gpu_memory_mb())

        if self.retriever is None:
            raise RuntimeError("Retrieval modes need a Retriever - build the "
                               "index first (ingest.py, embed_index.py).")

        hits = self.retriever.retrieve(question, top_k=top_k, mode="hybrid",
                                       use_reranker=use_reranker,
                                       expand_parents=expand_parents,
                                       refusal_threshold=refusal_threshold)
        if not hits:
            if mode == "strict":       # retrieval-level refusal: no LLM call
                return QAResult(question, cfg.refusal_text, mode, refused=True)
            hits = self.retriever.retrieve(question, top_k=top_k,
                                           mode="hybrid",
                                           use_reranker=use_reranker,
                                           expand_parents=expand_parents,
                                           refusal_threshold=0.0)

        context = build_context(hits)
        ctx_blocks = [r.parent_text or r.chunk.text for r in hits]
        sys, usr = build_prompt(question, context, mode, cfg=cfg)
        res = self.client.generate(sys, usr, temperature=GROUNDED_TEMPERATURE)
        result = QAResult(question, res.text, mode,
                          is_refusal(res.text, cfg), retrieved=hits, llm=res,
                          vram_mb=gpu_memory_mb())

        # faithfulness gate (strict mode only)
        if (use_gate and mode == "strict" and self.checker is not None
                and not result.refused):
            result.faithfulness = self.checker.report(result.answer, ctx_blocks)
            if (result.faithfulness["hallucination_rate"] > 0
                    and cfg.faithfulness_max_regens > 0):
                try:
                    sys2, usr2 = build_prompt(question, context, mode,
                                              corrective=True, cfg=cfg)
                    res2 = self.client.generate(sys2, usr2,
                                                temperature=GROUNDED_TEMPERATURE)
                    if res2.text.strip():
                        f2 = self.checker.report(res2.text, ctx_blocks)
                        if f2["hallucination_rate"] < \
                                result.faithfulness["hallucination_rate"]:
                            result.answer, result.llm = res2.text.strip(), res2
                            result.refused = is_refusal(res2.text, cfg)
                            result.faithfulness = f2
                except Exception as exc:
                    log.warning("Corrective regeneration failed (%s) - "
                               "keeping the original answer.", exc)
                result.regenerated = True
                result.flagged = result.faithfulness["hallucination_rate"] > 0

        if not result.answer.strip() and not result.refused:
            log.warning("Empty answer for %r — recording as abstention.",
                       question[:60])
            result.answer = cfg.refusal_text
            result.refused = True
            result.flagged = True
        return result

def load_pipeline(cfg: Config = CONFIG, model: str | None = None,
                  allow_fallback: bool = True) -> Pipeline:
    from embed_index import Embedder, HnswIndex
    from store import Store
    store = Store(cfg)
    if store.n_chunks() == 0:
        raise RuntimeError(f"Empty corpus DB at {cfg.db_path} - run "
                           f"python ingest.py first.")
    embedder = Embedder(cfg)
    index = HnswIndex.load(cfg)
    retriever = Retriever(store, index, embedder, cfg=cfg)
    client = OllamaClient(cfg, model=model, allow_fallback=allow_fallback)
    client.ensure_ready()

    import requests as _rq
    try:
        _rq.post(f"{client.base}/api/generate",
                 json={"model": client.model, "keep_alive": "24h",
                       "prompt": ""}, timeout=30)
    except Exception:
        pass

    checker = FaithfulnessChecker(cfg, embedder=embedder)
    return Pipeline(retriever, client, checker, cfg)


#self-check
def self_check(cfg: Config = CONFIG) -> bool:
    rows: list[tuple[str, bool, str]] = []

    def chk(name, fn):
        try:
            rows.append((name, True, str(fn() or "")[:60]))
        except Exception as exc:
            rows.append((name, False, f"{type(exc).__name__}: {exc}"[:60]))

    state: dict = {}
    def _load():
        state["pipe"] = load_pipeline(cfg)
        return f"{state['pipe'].retriever.store.n_chunks()} chunks"
    chk("Pipeline loads (DB+HNSW+Ollama)", _load)

    def _retrieval():
        pipe = state["pipe"]
        sample = pipe.retriever.store.sample_chunks(1)[0]
        hits = pipe.retriever.retrieve(sample.text[:120],
                                       refusal_threshold=0.0)
        assert hits, "no hits for an in-corpus sentence"
        state["sample"] = sample
        return f"top: {hits[0].chunk_id} (rerank={hits[0].rerank_score:.2f})"
    chk("Two-stage retrieval finds in-corpus text", _retrieval)

    def _grounded():
        r = state["pipe"].answer(
            f"What do the materials say about: "
            f"{state['sample'].text[:100]}?")
        assert r.answer
        f = r.faithfulness or {}
        return (f"refused" if r.refused else
                f"faith={f.get('faithfulness', -1):.2f} "
                f"({f.get('checker', '?')})")
    chk("Grounded answer + faithfulness gate", _grounded)

    chk("Out-of-scope refusal", lambda: (
        "refused" if state["pipe"].answer(
            "What is the capital of the planet Kepler-442b?").refused
        else (_ for _ in ()).throw(AssertionError("answered!"))))

    w = max(len(n) for n, _, _ in rows) + 2
    print("\nSELF-CHECK"); print("-" * (w + 50))
    ok = True
    for n, p, d in rows:
        ok &= p
        print(f"{n:<{w}} {'PASS' if p else 'FAIL':<6} {d}")
    print("-" * (w + 50)); print("OVERALL:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", default="strict",
                    choices=["strict", "normal", "baseline"])
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        raise SystemExit(0 if self_check() else 1)
    if not args.question:
        ap.error("Provide a question or --self-check")
    r = load_pipeline().answer(args.question, mode=args.mode)
    print(r.answer)
    if r.retrieved:
        print("\nSources:", ", ".join(x.chunk.citation for x in r.retrieved))
    if r.faithfulness:
        print(f"Faithfulness: {r.faithfulness['faithfulness']:.2f} "
              f"({r.faithfulness['checker']})"
              + (" [FLAGGED]" if r.flagged else ""))