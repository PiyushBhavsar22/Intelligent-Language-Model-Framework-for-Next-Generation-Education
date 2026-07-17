from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from config import CONFIG, Config
from llm import OllamaClient
from prompt import STRICT_SYSTEM
from store import Store, ChunkRow

log = logging.getLogger("raft")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

QGEN_PROMPT = """Read this excerpt from university module materials:

EXCERPT {citation}:
{text}

Write ONE exam-style question answerable ONLY from this excerpt, then a
grounded answer. The answer must: (1) begin with one short reasoning sentence
starting "Based on the materials, ", (2) use only facts from the excerpt,
(3) end with the citation {citation}.
Format exactly:
QUESTION: <question>
ANSWER: <answer>"""

OFF_TOPIC = [
    "Who won the 1998 FIFA World Cup?", "What is the boiling point of mercury?",
    "Summarise the plot of Hamlet.", "When did the French Revolution begin?",
    "How do I make sourdough bread?", "What is the population of Brazil?",
    "Who painted the Mona Lisa?", "What is the average rainfall in the Amazon?",
    "Explain photosynthesis in desert plants.", "Name three moons of Jupiter.",
]

_QA_RE = re.compile(r"QUESTION:\s*(.+?)\s*ANSWER:\s*(.+)", re.S)


def _context_block(chunks: list[ChunkRow]) -> str:
    return "\n\n".join(
        f"--- SOURCE {c.citation} (id={c.chunk_id}) ---\n{c.text}"
        for c in chunks)


def _example(question: str, context: str, target: str,
             cfg: Config) -> dict:
    return {"messages": [
        {"role": "system",
         "content": STRICT_SYSTEM.format(refusal=cfg.refusal_text)},
        {"role": "user",
         "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
        {"role": "assistant", "content": target}]}


def _stratified_pool(store: Store, n: int, used: set[str],
                     cfg: Config) -> list[ChunkRow]:
    """Round-robin across source files so every file gets at least one
    oracle before any file gets a second. Uniform random sampling over
    chunks silently starves small files in a fragmented corpus (see module
    docstring) - this is the fix.
    """
    all_chunks = [c for c in store.sample_chunks(10 ** 9)
                 if c.chunk_id not in used]
    by_source: dict[str, list[ChunkRow]] = {}
    for c in all_chunks:
        by_source.setdefault(c.source, []).append(c)

    rng = random.Random(cfg.seed)
    for chunks in by_source.values():
        rng.shuffle(chunks)

    buckets = list(by_source.values())
    pool: list[ChunkRow] = []
    i = 0
    while len(pool) < n and buckets:
        bucket = buckets[i % len(buckets)]
        if bucket:
            pool.append(bucket.pop())
            i += 1
        else:
            buckets.pop(i % len(buckets))
    return pool[:n]


def generate(cfg: Config = CONFIG, n: int | None = None,
             model: str | None = None) -> int:
    rng = random.Random(cfg.seed)
    store = Store(cfg)
    client = OllamaClient(cfg, model=model or cfg.model_name)
    client.ensure_ready()
    cfg.ensure_dirs()

    out_path = Path(cfg.raft_dir) / "raft_train.jsonl"
    used_path = Path(cfg.raft_dir) / "used_ids.txt"
    used: set[str] = set(used_path.read_text().splitlines()) \
        if used_path.exists() else set()

    n = n or cfg.raft_n_questions
    pool = _stratified_pool(store, n, used, cfg)
    if not pool:
        log.info("No unused chunks left - dataset complete.")
        return 0
    n_sources = len({c.source for c in pool})
    log.info("Generating %d RAFT examples with %s across %d distinct "
             "source files (resumable)...", len(pool), client.model,
             n_sources)

    written = 0
    with out_path.open("a", encoding="utf-8") as out, \
            used_path.open("a", encoding="utf-8") as used_f:
        for i, oracle in enumerate(pool):
            try:
                res = client.generate(
                    system="", user=QGEN_PROMPT.format(
                        citation=oracle.citation, text=oracle.text[:1800]),
                    temperature=0.7, max_new_tokens=1200)
            except Exception as exc:
                log.warning("QGen failed on %s: %s", oracle.chunk_id, exc)
                continue
            m = _QA_RE.search(res.text)
            if not m:
                continue
            question, answer = m.group(1).strip(), m.group(2).strip()
            if oracle.citation not in answer:      # enforce citation format
                answer = f"{answer} {oracle.citation}"

            distractors = [c for c in store.sample_chunks(
                cfg.raft_distractors + 4)
                if c.parent_id != oracle.parent_id][:cfg.raft_distractors]

            drop_oracle = rng.random() < cfg.raft_oracle_drop_frac
            docs = list(distractors) if drop_oracle \
                else [oracle] + list(distractors)
            rng.shuffle(docs)
            target = cfg.refusal_text if drop_oracle else answer
            out.write(json.dumps(
                _example(question, _context_block(docs), target, cfg),
                ensure_ascii=False) + "\n")
            used_f.write(oracle.chunk_id + "\n")
            written += 1
            if written % 25 == 0:
                log.info("  %d/%d", written, len(pool))

        remaining = _stratified_pool(store, 1, used | {c.chunk_id for c in pool}, cfg)
        if not remaining:
            total_written = sum(1 for _ in out_path.open(encoding="utf-8"))
            n_refusals = max(int(total_written * cfg.raft_refusal_frac), 5)
            log.info("Oracle pool exhausted - adding %d refusal examples "
                     "(final pass).", n_refusals)
            for q in rng.choices(OFF_TOPIC, k=n_refusals):
                docs = store.sample_chunks(cfg.raft_distractors + 1)
                out.write(json.dumps(
                    _example(q, _context_block(docs), cfg.refusal_text, cfg),
                    ensure_ascii=False) + "\n")
                written += 1
        else:
            log.info("Oracle pool not yet exhausted - skipping refusal "
                     "examples this run (they are added once, on the final "
                     "pass, to avoid compounding across resumed runs).")

    log.info("Wrote %d examples -> %s. Upload raft/ to Colab and run "
             "colab_raft_finetune.ipynb.", written, out_path)
    store.close()
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    generate(n=args.n, model=args.model)