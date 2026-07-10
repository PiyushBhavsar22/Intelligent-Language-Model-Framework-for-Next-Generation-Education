"""
Step 8 (optional, RQ2/RQ5): LoRA adapter fine-tuning on module data only.

Prereqs:
  * index built (scripts/1) and QA set filled in (scripts/2) — the training
    examples are derived from YOUR qa.jsonl references + retrieved context,
    so no external data touches the model.
  * extra packages:  pip install peft datasets
  * a GPU is strongly recommended (QLoRA on a T4 works; CPU is only for
    smoke-testing with 1 epoch and a tiny model).

Run:
    python scripts/5_finetune_lora.py            # build data + train (3 epochs)
    python scripts/5_finetune_lora.py --epochs 5

Then evaluate the fine-tuned condition:
    python scripts/3_run_experiments.py --adapter data/processed/lora_adapter
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetune import build_training_data, train_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    n = build_training_data()
    if n == 0:
        print("No usable QA items (each needs a non-empty 'reference'). "
              "Fill data/qa/qa.jsonl first.")
        sys.exit(1)

    adapter = train_lora(epochs=args.epochs, lr=args.lr, rank=args.rank)
    print(f"\nDone. Evaluate with:\n"
          f"  python scripts/3_run_experiments.py --adapter {adapter}")


if __name__ == "__main__":
    main()
