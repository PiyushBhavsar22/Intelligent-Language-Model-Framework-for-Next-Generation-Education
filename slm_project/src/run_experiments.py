from __future__ import annotations
import silence 
import argparse
import itertools
import logging
from pathlib import Path
import pandas as pd
from config import CONFIG, Config
from evaluate import (load_gold, relevant_flags, retrieval_metrics,
                      span_coverage, score_answer)

log = logging.getLogger("experiments")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _save(df: pd.DataFrame, name: str, cfg: Config) -> pd.DataFrame:
    cfg.ensure_dirs()
    out = Path(cfg.results_dir) / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== {name} -> {out} ===")
    with pd.option_context("display.width", 170, "display.max_columns", 24):
        print(df.round(3).to_string(index=False))
    return df


def _assert_single_model(df: pd.DataFrame, name: str) -> None:
    """Refuse to persist a mixed-model experiment. A silent OOM fallback
    mid-run swaps the generator and contaminates every subsequent row -
    the resulting averages describe no model at all. Fail loudly instead."""
    if "model" not in df.columns:
        return
    models = df["model"].dropna().unique()
    if len(models) > 1:
        counts = df["model"].value_counts().to_dict()
        raise RuntimeError(
            f"{name}: MIXED MODELS DETECTED {counts} - a VRAM OOM fallback "
            f"occurred mid-run. Results NOT saved. Free GPU memory (set "
            f"SLMQA_RERANKER_DEVICE=cpu, close other GPU apps) and re-run.")


def _condition_summary(df: pd.DataFrame, name: str, cfg: Config,
                       by: str = "condition") -> None:
    """Print + save a per-condition summary that does NOT hide structure:
    answerable and unanswerable items are aggregated separately, and the
    count of rows with missing faithfulness (refusals / retrieval-level
    short-circuits) is reported instead of silently averaged over. Flat
    means over a bimodal column is how the HHEM-truncation artifact hid
    for three experiment rounds - this makes that class of problem visible
    in the output itself."""
    if by not in df.columns:
        return
    metric_cols = [c for c in ["rouge_l", "bleu", "token_f1", "cosine_sim",
                               "faithfulness", "hallucination_rate",
                               "refusal_correct", "latency_s"]
                   if c in df.columns]
    parts = []
    for cond, sub in df.groupby(by):
        for answerable, tag in [(False, "answerable"), (True, "unanswerable")]:
            grp = sub[sub["unanswerable"] == answerable]                 if "unanswerable" in sub.columns else sub
            if grp.empty:
                continue
            row = {by: cond, "items": tag, "n": len(grp)}
            for m in metric_cols:
                row[m] = grp[m].mean()
            if "faithfulness" in grp.columns:
                row["n_missing_faith"] = int(grp["faithfulness"].isna().sum())
            parts.append(row)
    summary = pd.DataFrame(parts)
    _save(summary.round(3), f"{name}_summary", cfg)


def _retriever(cfg: Config):
    from embed_index import Embedder, HnswIndex
    from retriever import Retriever
    from store import Store
    return Retriever(Store(cfg), HnswIndex.load(cfg), Embedder(cfg), cfg=cfg)


def _safe_score(pipe, g: dict, cfg: Config, retries: int = 2,
                **answer_kwargs) -> dict | None:
    """Call pipe.answer + score_answer with retry-then-skip resilience.

    One bad/oversized prompt or a transient Ollama hiccup must never kill an
    hour-long experiment run. Retries a failed question up to `retries`
    times (Ollama sometimes recovers on the next call), then logs exactly
    which question was skipped and returns None so the caller can continue.
    """
    import time
    emb = pipe.retriever.embedder if pipe.retriever else None
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):          # e.g. retries=2 -> 3 tries
        try:
            res = pipe.answer(g["question"], **answer_kwargs)
            return score_answer(res, g, checker=pipe.checker, embedder=emb,
                                cfg=cfg)
        except Exception as exc:
            last_exc = exc
            if attempt <= retries:
                log.warning("Attempt %d/%d failed on %r (%s) - retrying...",
                           attempt, retries + 1, g["question"][:60], exc)
                time.sleep(3)
            continue
    log.error("SKIPPED after %d attempts: %r -> %s",
              retries + 1, g["question"][:60], last_exc)
    return None


def _run_scored(pipe, gold: list[dict], cfg: Config,
                extra: dict | None = None, **answer_kwargs) -> list[dict]:
    rows, skipped = [], 0
    for g in gold:
        row = _safe_score(pipe, g, cfg, **answer_kwargs)
        if row is None:
            skipped += 1
            continue
        rows.append(row | (extra or {}))
    if skipped:
        log.warning("%d/%d gold items skipped this condition (see errors "
                   "above).", skipped, len(gold))
    return rows


# exp1
def exp1_retrieval(cfg: Config = CONFIG) -> pd.DataFrame:
    gold = [g for g in load_gold(cfg) if not g["unanswerable"]]
    retr = _retriever(cfg)
    rows = []
    for mode, use_rr, k in itertools.product(
            ["dense", "lexical", "hybrid"], [False, True], [3, 5, 10]):
        agg: list[dict] = []
        for g in gold:
            hits = retr.retrieve(g["question"], top_k=k, mode=mode,
                                 use_reranker=use_rr, expand_parents=False,
                                 refusal_threshold=0.0)
            texts = [h.chunk.indexed_text for h in hits]
            flags = relevant_flags(texts, g["gold_spans"], cfg)
            m = retrieval_metrics(flags, len(g["gold_spans"]), k)
            m["span_coverage"] = span_coverage(texts, g["gold_spans"], cfg)
            agg.append(m)
        df = pd.DataFrame(agg).mean()
        rows.append({"pipeline": mode,
                     "reranker": use_rr, "top_k": k, **df.to_dict()})
    out = _save(pd.DataFrame(rows), "exp1_retrieval_ablation", cfg)
    best_r10 = out.loc[out.top_k == 10, "recall@10"].max() \
        if "recall@10" in out else 0.0
    branch = ("retrieval is the bottleneck -> run contextualize.py, consider "
              "embedder fine-tuning" if best_r10 < 0.7 else
              "retrieval healthy -> focus on generation (exp2/RAFT)")
    print(f"\nStage-0 decision: best recall@10 = {best_r10:.2f} -> {branch}")
    return out

# exp2-4
def _pipeline(cfg: Config, model: str | None):
    cfg.reranker_device = "cpu"
    from generate import load_pipeline
    return load_pipeline(cfg, model=model, allow_fallback=False)


def exp2_generation_ablation(cfg: Config = CONFIG,
                             model: str | None = None) -> pd.DataFrame:
    gold = load_gold(cfg)
    pipe = _pipeline(cfg, model)
    conditions = [("full (rerank+parents+gate)", True, True, True),
                  ("no-gate", True, True, False),
                  ("no-parents", True, False, True),
                  ("no-reranker", False, True, True)]
    rows = []
    for name, rr, par, gate in conditions:
        rows += _run_scored(pipe, gold, cfg, extra={"condition": name},
                            mode="strict", use_reranker=rr,
                            expand_parents=par, use_gate=gate)
    if not rows:
        raise RuntimeError("Every question failed in exp2 - check Ollama "
                           "(`ollama ps`) and the errors logged above.")
    df = pd.DataFrame(rows)
    _assert_single_model(df, "exp2_generation_ablation")
    _condition_summary(df, "exp2_generation_ablation", cfg)
    return _save(df, "exp2_generation_ablation", cfg)

def exp3_constraints(cfg: Config = CONFIG,
                     model: str | None = None) -> pd.DataFrame:
    gold = load_gold(cfg)
    pipe = _pipeline(cfg, model)
    conditions = [("normal-rag", "normal", False),
                  ("strict-prompt", "strict", False),
                  ("strict+gate", "strict", True)]
    rows = []
    for name, mode, gate in conditions:
        rows += _run_scored(pipe, gold, cfg,
                            extra={"condition": name, "model": pipe.client.model},
                            mode=mode, use_gate=gate)
    if not rows:
        raise RuntimeError("Every question failed in exp3 - check Ollama "
                           "(`ollama ps`) and the errors logged above.")
    df = pd.DataFrame(rows)
    _assert_single_model(df, "exp3_constraints")
    _condition_summary(df, "exp3_constraints", cfg)
    return _save(df, "exp3_constraints", cfg)

def exp4_baseline(cfg: Config = CONFIG,
                  model: str | None = None) -> pd.DataFrame:
    gold = load_gold(cfg)
    pipe = _pipeline(cfg, model)
    rows = []
    for mode in ["strict", "baseline"]:
        rows += _run_scored(pipe, gold, cfg, mode=mode)
    if not rows:
        raise RuntimeError("Every question failed in exp4 - check Ollama "
                           "(`ollama ps`) and the errors logged above.")
    df = pd.DataFrame(rows)
    _assert_single_model(df, "exp4_baseline")
    _condition_summary(df, "exp4_baseline", cfg, by="mode")
    return _save(df, "exp4_baseline", cfg)

def exp5_models(cfg: Config = CONFIG,
                model: str | None = None) -> pd.DataFrame:
    gold = load_gold(cfg)
    pipe = _pipeline(cfg, model)
    rows = _run_scored(pipe, gold, cfg, mode="strict")
    if not rows:
        raise RuntimeError("Every question failed in exp5 - check Ollama "
                           "(`ollama ps`) and the errors logged above.")
    df = pd.DataFrame(rows)
    _assert_single_model(df, "exp5")
    tag = (pipe.client.model.replace(":", "_").replace("/", "_"))
    _save(df, f"exp5_{tag}", cfg)
    parts = [pd.read_csv(f) for f in
             sorted(Path(cfg.results_dir).glob("exp5_*.csv"))
             if "summary" not in f.name]
    merged = pd.concat(parts)
    rq3_cols = [c for c in ["rouge_l", "token_f1", "cosine_sim",
                            "faithfulness", "hallucination_rate",
                            "refusal_correct", "latency_s", "tokens_per_s",
                            "vram_mb"] if c in merged.columns]
    summary = (merged.groupby("model")[rq3_cols]
               .mean().round(3).reset_index()
               .sort_values("vram_mb"))
    n_per_model = merged.groupby("model").size().rename("n_questions")
    summary = summary.merge(n_per_model, on="model")
    return _save(summary, "exp5_models_summary", cfg)

EXPS = {"1": exp1_retrieval, "2": exp2_generation_ablation,
        "3": exp3_constraints, "4": exp4_baseline, "5": exp5_models}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "1", "2", "3", "4", "5"])
    ap.add_argument("--model", default=None,
                    help="Override Ollama model (exp2-5)")
    args = ap.parse_args()
    todo = list(EXPS) if args.exp == "all" else [args.exp]
    for key in todo:
        fn = EXPS[key]
        fn(model=args.model) if key != "1" else fn()