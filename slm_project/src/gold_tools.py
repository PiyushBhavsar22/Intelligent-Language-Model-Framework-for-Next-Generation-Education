from __future__ import annotations
import argparse
import json
from pathlib import Path
from config import CONFIG, Config

def _append(item: dict, cfg: Config) -> None:
    cfg.ensure_dirs()
    with Path(cfg.gold_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def migrate(old_qa: Path, old_chunks: Path | None,
            cfg: Config = CONFIG) -> int:
    id_to_text: dict[str, str] = {}
    if old_chunks and old_chunks.exists():
        for line in old_chunks.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                id_to_text[c["chunk_id"]] = c["text"]
        print(f"Loaded {len(id_to_text)} v1 chunks for id->span lookup.")
    else:
        print("WARNING: no v1 chunk store - falling back to reference "
              "answers as provisional spans (marked needs_review).")

    n = 0
    for line in old_qa.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        spans, needs_review = [], False
        for cid in old.get("gold_chunk_ids", []):
            text = id_to_text.get(cid)
            if text:
                spans.append(" ".join(text.split()[:40]))
        if not spans:
            spans = [old["reference_answer"][:300]]
            needs_review = True
        item = {"question": old["question"],
                "reference_answer": old["reference_answer"],
                "gold_spans": spans, "unanswerable": False}
        if needs_review:
            item["needs_review"] = True
        _append(item, cfg)
        n += 1
    flagged = "" if id_to_text else " (ALL flagged needs_review - fix them!)"
    print(f"Migrated {n} items -> {cfg.gold_path}{flagged}")
    return n


def author(question: str, cfg: Config = CONFIG) -> None:
    from embed_index import Embedder, HnswIndex
    from retriever import Retriever
    from store import Store
    retr = Retriever(Store(cfg), HnswIndex.load(cfg), Embedder(cfg), cfg=cfg)
    hits = retr.retrieve(question, top_k=8, refusal_threshold=0.0)
    if not hits:
        print("Nothing retrieved - is the index built?")
        return
    for i, h in enumerate(hits):
        rs = f"{h.rerank_score:.2f}" if h.rerank_score is not None else "-"
        print(f"\n[{i}] {h.chunk_id}  rerank={rs}")
        print("    " + h.chunk.text[:400].replace("\n", " "))
    print("\nCopy a VERBATIM excerpt from the right chunk(s) as --span "
          "arguments to `gold_tools.py add`.")


def add(question: str, answer: str, spans: list[str], unanswerable: bool,
        cfg: Config = CONFIG) -> None:
    if not unanswerable and not spans:
        raise SystemExit("Answerable items need at least one --span.")
    _append({"question": question, "reference_answer": answer,
             "gold_spans": spans, "unanswerable": unanswerable}, cfg)
    print(f"Added. Gold set: {cfg.gold_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("migrate")
    m.add_argument("old_qa", type=Path)
    m.add_argument("--old-chunks", type=Path, default=None)

    a = sub.add_parser("author")
    a.add_argument("question")

    d = sub.add_parser("add")
    d.add_argument("--question", required=True)
    d.add_argument("--answer", default="")
    d.add_argument("--span", action="append", default=[])
    d.add_argument("--unanswerable", action="store_true")

    args = ap.parse_args()
    if args.cmd == "migrate":
        migrate(args.old_qa, args.old_chunks)
    elif args.cmd == "author":
        author(args.question)
    else:
        add(args.question, args.answer, args.span, args.unanswerable)
