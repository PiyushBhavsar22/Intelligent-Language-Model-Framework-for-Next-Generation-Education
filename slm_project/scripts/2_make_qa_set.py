"""
Step 2: Create / seed the evaluation QA set.

The evaluation needs question/answer pairs. Each item has:
    question      : the query
    reference     : a gold reference answer (for ROUGE/BLEU/F1/cosine)
    gold_sources  : list of source filenames that contain the answer
                    (for precision@k / recall@k). Use the 'source' strings
                    exactly as they appear in data/processed/chunks.jsonl.

This script writes a small STARTER file you then expand by hand from your
tutorial/exam questions. It also prints the available sources so you can copy
the right filenames into gold_sources.

Run:
    python scripts/2_make_qa_set.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import QA_DIR, CHUNKS_FILE

QA_FILE = QA_DIR / "qa.jsonl"

STARTER = [
    {
        "question": "What is the CIA triad in computer security?",
        "reference": ("The CIA triad refers to Confidentiality, Integrity and "
                      "Availability, the three core objectives of information security."),
        "gold_sources": [
            "MOOC 1/Topic 1/M1.1.8 Confidentiality, integrity, availability (CIA) triad - Article.docx"
        ],
    },
    {
        "question": "What does the Bell-LaPadula model enforce?",
        "reference": ("The Bell-LaPadula model is a confidentiality model enforcing "
                      "no-read-up and no-write-down to protect classified information."),
        "gold_sources": [
            "MOOC 1/Topic 4/M1.4.3 Bell-LaPadula Model - Article.docx"
        ],
    },
    {
        "question": "How is a threat defined and classified in this module?",
        "reference": ("A threat is a potential cause of an unwanted incident that may "
                      "harm a system; threats are classified by their nature and source."),
        "gold_sources": [
            "MOOC 1/Topic 1/M1.1.11 Threats_ definition and classification - Article.docx"
        ],
    },
]


def main() -> None:
    if not CHUNKS_FILE.exists():
        print("Build the index first: python scripts/1_build_index.py")
        sys.exit(1)

    # show available sources to help you fill gold_sources correctly
    sources = sorted({json.loads(l)["source"] for l in open(CHUNKS_FILE, encoding="utf-8")})
    print(f"{len(sources)} sources available. First 25:")
    for s in sources[:25]:
        print("   ", s)

    if QA_FILE.exists():
        print(f"\n{QA_FILE} already exists — not overwriting.")
        return

    with open(QA_FILE, "w", encoding="utf-8") as f:
        for item in STARTER:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote starter QA set to {QA_FILE} ({len(STARTER)} items).")
    print("EDIT this file: add your own tutorial/exam questions and set "
          "gold_sources using the source strings printed above.")


if __name__ == "__main__":
    main()
