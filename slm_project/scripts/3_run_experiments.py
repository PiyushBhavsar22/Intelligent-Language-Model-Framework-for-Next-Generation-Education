"""
Step 3: Run the full experiment matrix and write results.

For every question in the QA set, this evaluates:
  * retrieval methods : dense / lexical / hybrid  (precision@k, recall@k)
  * constrained SLM   : answer using only retrieved context (grounding + similarity)
  * baseline LLM      : answer with no context      (grounding vs SAME context, similarity)
and records latency + a per-question CSV plus an aggregate summary.

Run:
    python scripts/3_run_experiments.py

Notes:
  * First run downloads the embedding + generator models (needs internet).
  * On CPU this is slow; on a T4 GPU it is fast. Reduce QA set / TOP_K to test.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QA_DIR, OUTPUTS_DIR, TOP_K
from src.retriever import Retriever
from src.generator import Generator
from src.memtrack import MemoryTracker, model_footprint_mb
from src.evaluation import (
    precision_recall_at_k, grounding_score, hallucination_rate,
    text_similarity_bundle,
)

QA_FILE = QA_DIR / "qa.jsonl"
RETRIEVAL_METHODS = ["dense", "lexical", "hybrid", "hierarchical"]


def load_qa() -> list[dict]:
    if not QA_FILE.exists():
        print("No QA set. Run: python scripts/2_make_qa_set.py")
        sys.exit(1)
    return [json.loads(l) for l in open(QA_FILE, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter dir (from scripts/5) to "
                         "evaluate the fine-tuned condition")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. --tag qwen1.5b "
                         "so runs with different models don't overwrite")
    args = ap.parse_args()

    qa = load_qa()
    print(f"Loaded {len(qa)} QA items.")

    retriever = Retriever()
    generator = Generator(adapter_path=args.adapter)
    footprint = model_footprint_mb()
    print(f"Model footprint after load: {footprint}")

    rows = []
    for qi, item in enumerate(qa, 1):
        q = item["question"]
        ref = item.get("reference", "")
        gold = set(item.get("gold_sources", []))
        print(f"\n[{qi}/{len(qa)}] {q}")

        # ---- retrieval metrics per method ----
        retrieved_by_method = {}
        for method in RETRIEVAL_METHODS:
            res = retriever.search(q, top_k=TOP_K, method=method)
            retrieved_by_method[method] = res
            srcs = [r["source"] for r in res]
            p, r_ = precision_recall_at_k(srcs, gold, k=TOP_K)
            rows.append({
                "question": q, "stage": "retrieval", "method": method,
                "precision_at_k": round(p, 4), "recall_at_k": round(r_, 4),
            })

        # use the HYBRID context for generation (best of both)
        contexts = retrieved_by_method["hybrid"]

        # ---- constrained SLM ----
        t0 = time.time()
        with MemoryTracker() as mem_c:
            ans_c = generator.generate_constrained(q, contexts)
        lat_c = time.time() - t0
        g_c = grounding_score(ans_c, contexts)
        sim_c = text_similarity_bundle(ans_c, ref) if ref else {}
        rows.append({
            "question": q, "stage": "generation", "method": "constrained_slm",
            "grounding": round(g_c, 4),
            "hallucination_rate": round(1 - g_c, 4),
            "latency_s": round(lat_c, 3),
            "peak_vram_mb": round(mem_c.peak_vram_mb, 1),
            "ram_mb": round(mem_c.ram_mb, 1),
            **sim_c,
            "answer": ans_c,
        })
        print(f"  constrained  grounding={g_c:.2f}  latency={lat_c:.1f}s")

        # ---- baseline (no context) ----
        t0 = time.time()
        with MemoryTracker() as mem_b:
            ans_b = generator.generate_baseline(q)
        lat_b = time.time() - t0
        # grounded against the SAME module context, to measure how much of the
        # baseline answer is actually supported by module material
        g_b = grounding_score(ans_b, contexts)
        sim_b = text_similarity_bundle(ans_b, ref) if ref else {}
        rows.append({
            "question": q, "stage": "generation", "method": "baseline_llm",
            "grounding": round(g_b, 4),
            "hallucination_rate": round(1 - g_b, 4),
            "latency_s": round(lat_b, 3),
            "peak_vram_mb": round(mem_b.peak_vram_mb, 1),
            "ram_mb": round(mem_b.ram_mb, 1),
            **sim_b,
            "answer": ans_b,
        })
        print(f"  baseline     grounding={g_b:.2f}  latency={lat_b:.1f}s")

    df = pd.DataFrame(rows)
    tag = f"_{args.tag}" if args.tag else ("_lora" if args.adapter else "")
    per_q = OUTPUTS_DIR / f"results_per_question{tag}.csv"
    df.to_csv(per_q, index=False)

    # ---- aggregate summary ----
    ret = df[df.stage == "retrieval"].groupby("method")[
        ["precision_at_k", "recall_at_k"]].mean().round(4)
    gen = df[df.stage == "generation"].groupby("method")[
        [c for c in ["grounding", "hallucination_rate", "latency_s",
                     "peak_vram_mb", "ram_mb",
                     "rougeL", "bleu", "token_f1", "cosine"] if c in df.columns]
    ].mean().round(4)

    summary_path = OUTPUTS_DIR / f"summary{tag}.txt"
    with open(summary_path, "w") as f:
        f.write(f"model: {generator.model_name}"
                f"{'  +LoRA(' + str(args.adapter) + ')' if args.adapter else ''}\n")
        f.write(f"model footprint after load: {footprint}\n\n")
        f.write("=== RETRIEVAL (mean over questions) ===\n")
        f.write(ret.to_string() + "\n\n")
        f.write("=== GENERATION (mean over questions) ===\n")
        f.write(gen.to_string() + "\n")

    print("\n" + "=" * 60)
    print("RETRIEVAL:\n", ret)
    print("\nGENERATION:\n", gen)
    print("=" * 60)
    print(f"\nWrote:\n  {per_q}\n  {summary_path}")


if __name__ == "__main__":
    main()
