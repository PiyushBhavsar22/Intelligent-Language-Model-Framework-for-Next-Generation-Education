"""
Step 4 (optional): Interactive Q&A against the constrained SLM.

Run:
    python scripts/4_ask.py
Type a question, get a grounded answer + the sources used. Ctrl-C to quit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TOP_K
from src.retriever import Retriever
from src.generator import Generator
from src.evaluation import grounding_score


def main() -> None:
    print("Loading retriever + model (first run downloads weights)...")
    retriever = Retriever()
    generator = Generator()
    print("Ready. Ask a question (Ctrl-C to quit).\n")

    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                continue
            ctx = retriever.search(q, top_k=TOP_K, method="hybrid")
            ans = generator.generate_constrained(q, ctx)
            g = grounding_score(ans, ctx)
            print(f"\nA: {ans}")
            print(f"\n[grounding={g:.2f}] sources used:")
            for i, c in enumerate(ctx, 1):
                print(f"  [{i}] {c['source']}  (score={c['score']:.3f})")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
