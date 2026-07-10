"""
Step 1: Extract -> clean -> chunk -> build FAISS + BM25 index.

Run from the PROJECT ROOT:
    python scripts/1_build_index.py

Prereq: your extracted module folders (e.g. 'MOOC 1', 'MOOC 2', ...) must be
inside data/raw/.
"""
import sys
from pathlib import Path

# make 'src' importable no matter where you run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR
from src.extract import iter_documents
from src.chunk import chunk_document
from src.retriever import build_index


def main() -> None:
    docs = list(iter_documents(RAW_DIR))
    if not docs:
        print(f"No documents found in {RAW_DIR}. "
              f"Did you unzip your MOOC folders into data/raw/ ?")
        sys.exit(1)

    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d))
    print(f"{len(docs)} documents -> {len(chunks)} chunks")

    build_index(chunks)
    print("Done. Index written to data/processed/")


if __name__ == "__main__":
    main()
