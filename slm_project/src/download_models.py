from __future__ import annotations
import os
import sys
os.environ["SLMQA_OFFLINE"] = "0"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from config import CONFIG  # noqa: E402


def main() -> int:
    print("One-time model download - needs internet. Everything after this "
          "runs fully offline.\n")

    print(f"[1/3] Embedder  {CONFIG.embed_model_name} ...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SentenceTransformer(CONFIG.embed_model_name, device="cpu")
    print("      cached OK\n")

    print(f"[2/3] Reranker  {CONFIG.reranker_model_name} ...")
    CrossEncoder(CONFIG.reranker_model_name, device="cpu",
                 max_length=CONFIG.rerank_max_length)
    print("      cached OK\n")

    print(f"[3/3] Faithfulness gate  {CONFIG.hhem_model_name} ...")
    try:
        from transformers import AutoModelForSequenceClassification
        AutoModelForSequenceClassification.from_pretrained(
            CONFIG.hhem_model_name, trust_remote_code=True)
        print("      cached OK\n")
    except Exception as exc:
        print(f"      WARNING: HHEM failed to download ({exc}).\n"
              f"      The pipeline still works - faithfulness.py falls back "
              f"to embedding-similarity grounding.\n")

    print("Done. Everything now runs fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())