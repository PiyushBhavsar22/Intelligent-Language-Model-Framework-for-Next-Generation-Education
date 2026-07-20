# Grounding Beats Scale: Small Language Models and Faithfulness Gating for Closed-Corpus Question Answering

**A closed-corpus, faithfulness-gated Small Language Model for university module question answering fully offline, single-GPU, and evaluated end-to-end.**

This system constrains a 3–8B parameter Small Language Model (SLM) to answer questions **strictly** from the materials of a single university module. It never falls back on parametric knowledge, it cites the exact source passage behind every claim, and it refuses when the module materials don't contain the answer all served locally on a single 8GB consumer GPU, with no network dependency at inference time.

The project treats hallucination as a retrieval and constraint problem rather than a *scale* problem: a small, tightly-grounded model answering only from verified course content is measured against a general-purpose LLM answering from parametric memory alone, across retrieval quality, groundedness, answer similarity, and on-device efficiency.

---

## Why This Exists

General-purpose LLMs are increasingly used by students for explanation and tutoring, but their fluency is exactly what makes them dangerous in coursework: an unsupported, confidently-stated answer is indistinguishable in tone from a correct one, and the learner by definition not yet expert in the material is poorly positioned to catch the difference. This system removes that risk at the architecture level rather than relying on the model to police itself: retrieval supplies the evidence, a strict prompt binds generation to that evidence, and a per-sentence faithfulness check gates the output before it ever reaches the student.

---

## Research Questions

| # | Question |
|---|---|
| RQ1 | How effectively can a constrained SLM retrieve, ground, and synthesize answers from a closed corpus? |
| RQ2 | Which constraint mechanism prompt-conditioning, retrieval-level filtering, or lightweight fine-tuning most effectively suppresses hallucinated or out-of-scope content? |
| RQ3 | How does model size (3–8B+) trade against accuracy, latency, and memory on consumer hardware? |
| RQ4 | Which retrieval strategy (dense, lexical, hybrid) maximizes relevance and minimizes noise? |
| RQ5 | Can adapter-based fine-tuning encode grounding behaviour without exposing the model to external data? |
| RQ6 | How does the constrained SLM compare to a general-purpose LLM on hallucination rate, retrieval precision, and efficiency? |

---

## System Architecture

The pipeline is split across two hosts by hard necessity, not preference: gradient-based fine-tuning exceeds the local 8GB VRAM budget, so adaptation is offloaded to a free-tier Colab T4, while every other stage ingestion, indexing, retrieval, generation, evaluation, and serving runs entirely on a local RTX 4060 laptop with the network physically disconnected after a one-time bootstrap.

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL HOST RTX 4060 (8GB VRAM), fully offline at inference    │
│                                                                   │
│  Ingestion → Chunking → Contextualization → Embedding/Index      │
│  (PDF/PPTX/DOCX/VTT/XLSX/TXT)   (parent/child)   (bge-m3 + FAISS) │
│                                                                   │
│  Query → Hybrid Retrieval → Rerank → Constrained Generation      │
│          (dense+BM25, RRF)   (cross-encoder)   (Ollama, strict)  │
│                                            │                     │
│                                  Faithfulness Gate (HHEM-2.1)     │
│                                  → corrective regen → answer      │
│                                                                   │
│  Span-Based Evaluation Harness → 5 ablation suites → results/    │
│  Streamlit Interface (student-facing, per-answer verification)   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │  merged Q4_K_M GGUF, one-time upload
                              │
┌─────────────────────────────────────────────────────────────────┐
│  COLAB HOST free-tier T4 (16GB), training only                 │
│  RAFT dataset (local) → Unsloth QLoRA fine-tune → GGUF export     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### 1. Ingestion `ingest.py`
Parses the module corpus across six file formats through a type-dispatched parser registry: PDF (four-tier fallback chain `pymupdf4llm` → `PyMuPDF` → `pypdf` → `pdfminer.six`, richest-to-most-compatible), DOCX (heading-sectioned via `python-docx`), PPTX (text frames, tables, and speaker notes), VTT (subtitle transcripts, timestamps stripped), XLSX (per-worksheet), and TXT. Ingestion is incremental a SHA-256 manifest means only new or changed files are re-parsed and files with non-text extensions are counted and logged rather than silently dropped, so the boundary of what the pipeline can and cannot index is always explicit.

### 2. Structure-Aware Chunking `chunk.py`
Documents are split into 320-token child chunks (48-token overlap) for precise retrieval, paired with 1200-token parent chunks handed to the generator for sufficient context. This decouples retrieval precision from generation context the two have opposite granularity needs, and a single chunk size can't satisfy both.

### 3. Contextual Retrieval Enrichment `contextualize.py`
Each child chunk is prepended with a locally-generated 50–100 token situating summary (document, topic, what the chunk covers) before embedding and BM25 indexing an isolated chunk reading "this is bounded by O(n log n)" is unretrievable without knowing which algorithm and lecture it belongs to. Fully resumable; only chunks missing context are processed on any given run.

### 4. Embedding & Indexing `embed_index.py`
Chunks are embedded with `BAAI/bge-m3` (1024-dim, no query-instruction prefix required) and cached as float16 BLOBs in SQLite, so only new/changed/invalidated rows are ever re-embedded. The FAISS HNSW graph is rebuilt from the cached matrix on every corpus change re-embedding is the expensive step and it's cached; rebuilding the graph completes in seconds.

### 5. Two-Stage Hybrid Retrieval `retriever.py`
**Stage 1 recall:** dense HNSW search (top-50) and BM25 over SQLite FTS5 (top-50), fused via weighted Reciprocal Rank Fusion (k=60, dense weight 0.6, lexical weight 0.4) rank-based fusion sidesteps the fact that cosine and BM25 scores live on incomparable scales. **Stage 2 precision:** the fused pool is reranked by `bge-reranker-v2-m3`, a cross-encoder that jointly attends over query and candidate; the top-5 survive. A pre-rerank dense floor and a post-rerank threshold enforce closed-corpus refusal at the retrieval level in strict mode, an out-of-scope query never reaches the LLM at all.

### 6. Constrained Generation `generate.py`, `llm.py`
Served through Ollama at temperature 0.3 (reduced sampling entropy for factual QA), `num_ctx=8192`, fixed seed. Three prompt modes constitute the constraint-mechanism comparison: **strict** (context-only, fixed refusal string otherwise), **normal** (context supplied, no explicit prohibition), and **baseline** (no retrieval, parametric memory only the general-LLM control). The Ollama client is defensive by necessity: daemon/model-availability checks, exponential backoff, VRAM-crash classification with automatic fallback, and empty completions treated as a first-class failure mode (abstention) rather than a silent success or a crash.

### 7. Faithfulness Gate `faithfulness.py`
Every answer is split into sentences and scored against the retrieved context by `Vectara HHEM-2.1-Open`, a 110M-parameter NLI cross-encoder premise and hypothesis are truncated with the model's own tokenizer to explicit token budgets, and each sentence is scored against every retrieved chunk (not one concatenated, truncated block), so evidence anywhere in the context is visible to the checker. Sentences scoring below threshold trigger one corrective regeneration; the replacement is accepted only if non-empty and strictly improved, so a degenerate empty response can never silently overwrite a good answer. Degrades gracefully to embedding-cosine grounding if HHEM is unavailable, with the checker identity always reported alongside the score.

### 8. Span-Based Evaluation `evaluate.py`
Gold items are anchored on **verbatim text spans** from the corpus, not chunk IDs a chunk is relevant if a gold span matches it via substring containment, token-recall ≥0.60, or fuzzy ratio ≥0.90. This makes the gold set invariant to re-chunking, which matters: an earlier identifier-keyed version of this harness silently zeroed every retrieval metric the moment the corpus was re-chunked, and it looked exactly like a model failure until traced to the harness itself.

### 9. Fine-Tuning Offload `raft_data.py` + Colab notebook
RAFT-style training data (question + one oracle chunk + distractors, oracle omitted in 20% of examples to teach refusal, stratified across source files so small documents aren't starved by uniform-random sampling) is generated locally and fine-tuned via Unsloth QLoRA on a free Colab T4. Loss is computed on the assistant response only masking prompt tokens prevents the adapter from memorizing retrieved context instead of learning grounding behaviour. The merged, quantized (Q4_K_M) GGUF returns to the local host and serves through Ollama indistinguishably from any other model the offline serving guarantee is never broken, since Colab only ever touches training text, never live inference traffic.

---

## Evaluation Suite

Five ablation suites, each mapped to specific research questions, executed via `run_experiments.py`:

| Suite | Compares | Answers |
|---|---|---|
| **Exp 1 Retrieval Ablation** | {dense, lexical, hybrid} × {reranker on/off} × k∈{3,5,10}, no LLM calls | RQ1, RQ4 |
| **Exp 2 Generation Ablation** | full pipeline vs. no-gate vs. no-parent-expansion vs. no-reranker | RQ1, RQ2 |
| **Exp 3 Constraint Mechanisms** | normal-RAG vs. strict-prompt vs. strict+gate vs. fine-tuned adapter | RQ2, RQ5 |
| **Exp 4 Constrained vs. Baseline** | strict (full pipeline) vs. no-retrieval parametric-only | RQ6 |
| **Exp 5 Model Size** | 1B / 4B / E4B / 12B / fine-tuned, identical gold set | RQ3 |

**Metric battery:** precision@k / recall@k / hit@k / MRR / nDCG@k / span coverage (retrieval); per-sentence HHEM faithfulness and hallucination rate, checker-tagged (grounding); ROUGE-L, BLEU-4, token-F1, embedding cosine (similarity reported but never treated as sufficient on its own, since fluent text can score well while remaining unfaithful); refusal correctness scored in **both** directions, so over-refusal is exposed as clearly as under-refusal; end-to-end latency, decode throughput, and whole-GPU VRAM via `nvidia-smi`.

Every run is seeded (`seed=42`); every configuration value lives in a single `config.py` dataclass, overridable via `SLMQA_*` environment variables an experiment is fully specified by its config plus CLI flags.

---

## Key Findings

- **Grounding beats scale.** Moving from unconstrained retrieval to a strict, faithfulness-gated prompt reduces hallucination further than moving from a 4B to a 12B model a small, tightly-grounded model outperforms a large, loosely-grounded one on this task.
- **The reranker earns its cost.** Precision at small k the regime that determines what the generator actually sees improves substantially with cross-encoder reranking; the effect is largest exactly where it matters most (k=5).
- **Refusal is a control variable, not padding.** A system optimized purely for low hallucination can trivially minimize it by refusing everything; scoring refusal correctness on both answerable and unanswerable items exposes this degenerate strategy rather than hiding it behind a good-looking faithfulness number.
- **A measurement bug can look exactly like a model failure.** A faithfulness checker with a context window smaller than the retrieved evidence silently penalized correctly-grounded answers hallucination rate dropped by roughly 4× once the checker was corrected to see all retrieved evidence rather than a truncated fragment of it, with zero change to the underlying model or prompts.
- **Model size is non-monotonic on this corpus.** Faithfulness peaks in the 4B–E4B range and *degrades* at 12B, which also pays a substantial latency penalty from partial CPU offload on an 8GB card bigger is not better once the model no longer fits comfortably on the deployment host.

---

## Repository Structure

```
config.py              single source of truth every tunable value, env-overridable
silence.py              suppresses cosmetic library noise across entry points

ingest.py                multi-format corpus parsing, incremental, parallelized
chunk.py                 parent/child structure-aware chunking
contextualize.py         situating-context enrichment (resumable, one LLM call/chunk)
embed_index.py           bge-m3 embedding + FAISS HNSW indexing (cached, incremental)
store.py                 SQLite-backed chunk store, embeddings, FTS5, manifest

retriever.py             two-stage hybrid retrieval (RRF fusion + cross-encoder rerank)
prompt.py                strict/normal/baseline prompt construction, refusal detection
llm.py                   defensive Ollama client (fallback chain, empty-response handling)
generate.py              end-to-end answer pipeline, faithfulness gate integration
faithfulness.py          HHEM-based per-sentence groundedness scoring

evaluate.py               span-based gold matching, full metric battery
gold_tools.py              gold-set authoring against the live index
check_gold.py              gold-set integrity check
check_coverage.py          corpus ingestion coverage diff (disk vs. index)
check_raft_coverage.py     RAFT oracle sampling coverage diff

raft_data.py               stratified RAFT training-data generation (local)
colab_raft_finetune.ipynb  Unsloth QLoRA fine-tune + GGUF export (Colab)

run_experiments.py         five ablation suites, purity-gated, resilient to failures

download_models.py         one-time online bootstrap (embedder, reranker, HHEM)
```

---

## Design Principles

- **Offline by construction, not by convention.** Every model-loading library inherits strict offline mode before it's ever imported; a single sanctioned online moment (`download_models.py`) is the only network dependency in the system's lifetime.
- **Every metric is checker-tagged.** Faithfulness scores never get silently averaged across HHEM and its embedding-cosine fallback the checker identity travels with every number.
- **Resumability as a default, not an afterthought.** Ingestion, contextualization, and RAFT generation all persist progress and pick up exactly where they left off a multi-hour job interrupted at 90% loses nothing.
- **A model swap mid-experiment is worse than a crash.** Every generation experiment pins its model explicitly; an out-of-memory event retries the same model rather than silently falling back to a different one, and a purity gate refuses to save any result set that isn't single-model.
- **Text, not identifiers, as the ground truth.** The gold evaluation set is anchored on verbatim corpus spans specifically so that re-chunking the corpus can never silently invalidate every retrieval metric a failure mode this project hit once and re-architected against permanently.

---

## Hardware & Stack

| | |
|---|---|
| **Inference host** | RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM, i7-13650HX |
| **Fine-tuning host** | Google Colab, free-tier T4 (16GB) |
| **Serving** | Ollama (local, fully offline) |
| **Embeddings / Reranking** | BAAI/bge-m3, bge-reranker-v2-m3, FAISS HNSW |
| **Faithfulness** | Vectara HHEM-2.1-Open |
| **Fine-tuning** | Unsloth QLoRA, RAFT-style data construction |

---

## License & Scope

This is a practicum research project. The evaluation corpus consists of licensed university module materials and is not distributed in this repository. Results are evidence about closed, curated, single-module corpora specifically not benchmark figures intended to transfer to open-domain question answering.
