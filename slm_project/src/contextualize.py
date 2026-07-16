from __future__ import annotations
import argparse
import logging
import time
from config import CONFIG, Config
from llm import OllamaClient
from store import Store

log = logging.getLogger("contextualize")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONTEXT_PROMPT = """<document>
{doc_head}
</document>

Here is a chunk from within that document:
<chunk>
{chunk}
</chunk>

Write a short context (2-3 sentences, max {max_tokens} tokens) situating this
chunk within the overall document for search retrieval: name the document /
lecture topic and what this specific chunk covers. Answer with ONLY the
context, nothing else."""

def contextualize(cfg: Config = CONFIG, limit: int | None = None,
                  model: str | None = None) -> int:
    store = Store(cfg)
    todo = store.chunks_missing_context(limit=limit)
    if not todo:
        log.info("All chunks already have contexts - nothing to do.")
        return 0

    client = OllamaClient(cfg, model=model or cfg.model_name)
    client.ensure_ready()
    log.info("Generating situating contexts for %d chunks with %s "
             "(resumable - safe to interrupt).", len(todo), client.model)

    doc_heads: dict[str, str] = {}
    done, t0 = 0, time.perf_counter()
    for row in todo:
        head = doc_heads.setdefault(
            row.source, store.doc_head(row.source,
                                       cfg.context_doc_snippet_chars))
        prompt = CONTEXT_PROMPT.format(doc_head=head, chunk=row.text[:2000],
                                       max_tokens=cfg.context_max_tokens)
        try:
            res = client.generate(system="", user=prompt, temperature=0.3,
                                  max_new_tokens=max(
                                      cfg.context_max_tokens * 4, 360))
            ctx = " ".join(res.text.split())[:600]
            if ctx:
                store.set_context(row.chunk_id, ctx)
                done += 1
        except Exception as exc:
            log.warning("Context failed for %s (%s) - will retry on next run.",
                        row.chunk_id, exc)
        if done and done % 50 == 0:
            rate = done / (time.perf_counter() - t0)
            eta_h = (len(todo) - done) / max(rate, 1e-9) / 3600
            log.info("%d/%d done (%.1f/s, ~%.1f h remaining)",
                     done, len(todo), rate, eta_h)

    log.info("Contextualized %d chunks. Re-run `python embed_index.py` to "
             "re-embed the updated rows.", done)
    store.close()
    return done

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N chunks this run")
    ap.add_argument("--model", default=None,
                    help="Ollama model for contextualizing (default gemma4:e4b)")
    args = ap.parse_args()
    contextualize(limit=args.limit, model=args.model)
