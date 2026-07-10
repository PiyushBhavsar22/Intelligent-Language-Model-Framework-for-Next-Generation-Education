# Constrained SLM for a Single University Module (RAG)

A research pipeline that answers questions **strictly from one module's own
materials** (lecture notes, slides, transcripts, references) using a small
open-source language model + retrieval-augmented generation. It measures
retrieval quality, grounding/hallucination, answer similarity, and efficiency —
exactly the metrics in the practicum proposal.

```
slm_project/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py        # all paths + hyperparameters (edit here)
│   ├── extract.py       # docx / pdf / vtt -> clean text
│   ├── chunk.py         # overlapping chunks
│   ├── retriever.py     # FAISS (dense) + BM25 (lexical) + hybrid + hierarchical routing
│   ├── generator.py     # SLM loading (+optional LoRA adapter) + constrained/baseline prompting
│   ├── evaluation.py    # precision@k, recall@k, grounding, ROUGE/BLEU/F1/cosine
│   ├── memtrack.py      # peak VRAM + process RAM measurement (RQ3)
│   └── finetune.py      # LoRA/QLoRA adapter fine-tuning on module data (RQ2/RQ5)
├── scripts/
│   ├── 0_prepare_data.py   # unzip MOOC_*.zip into data/raw/
│   ├── 1_build_index.py    # build the retrieval index
│   ├── 2_make_qa_set.py    # seed the evaluation Q&A set
│   ├── 3_run_experiments.py# full experiment matrix -> outputs/ (--adapter, --tag)
│   ├── 4_ask.py            # interactive Q&A (optional)
│   └── 5_finetune_lora.py  # train a LoRA adapter, then re-run experiments with it
└── data/
    ├── raw/         <-- put your extracted "MOOC 1", "MOOC 2", ... here
    ├── processed/   (auto: chunks + FAISS + BM25)
    └── qa/          (your evaluation questions)
```

---

## Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
cd slm_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
cd slm_project
python3 -m venv .venv
source .venv/bin/activate
```

Use **Python 3.10–3.12**. Check with `python --version`.

---

## Step 2 — Install PyTorch FIRST (this avoids 90% of setup errors)

Pick the command matching your machine from https://pytorch.org/get-started/locally/ .
Common cases:

**NVIDIA GPU (CUDA 12.1) — recommended for the T4:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (laptop, no GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Do NOT skip this and let `requirements.txt` pull torch — installing torch
> explicitly with the right CUDA build is what prevents the classic
> "CUDA not available" / "torch built without CUDA" problems.

---

## Step 3 — Install the rest

```bash
pip install -r requirements.txt
```

`bitsandbytes` (4-bit quantization) only installs on Linux+NVIDIA. If it fails
on Windows/Mac/CPU, **ignore it** — the code auto-falls-back to normal
precision. To force-skip it: delete its line from `requirements.txt`.

---

## Step 4 — Add your data

You have `MOOC_1.zip`, `MOOC_2.zip`, … Put them in `data/raw/` and run:

```bash
python scripts/0_prepare_data.py
```

This unzips them into `data/raw/MOOC 1`, `data/raw/MOOC 2`, … You do **not**
need to delete images or spreadsheets — the extractor only reads `.docx`,
`.pdf`, and `.vtt` and ignores everything else.

(If you already extracted the folders, just make sure they sit directly under
`data/raw/`.)

---

## Step 5 — Build the retrieval index

```bash
python scripts/1_build_index.py
```

First run downloads the embedding model (~80 MB). Produces
`data/processed/chunks.jsonl`, `faiss.index`, `bm25.pkl`.

---

## Step 6 — Create the evaluation Q&A set

```bash
python scripts/2_make_qa_set.py
```

This writes a **starter** `data/qa/qa.jsonl` (3 example questions) and prints
the available source filenames. **Open that file and add your own
tutorial/exam questions.** Each line is one JSON object:

```json
{"question": "…", "reference": "gold answer text", "gold_sources": ["MOOC 1/Topic 4/M1.4.3 Bell-LaPadula Model - Article.docx"]}
```

- `reference` powers ROUGE/BLEU/F1/cosine.
- `gold_sources` powers precision@k / recall@k — copy the exact `source`
  strings printed by the script.

Aim for 20–40 questions for a meaningful evaluation.

---

## Step 7 — Run the experiments

```bash
python scripts/3_run_experiments.py
```

For every question it compares **dense vs lexical vs hybrid vs hierarchical**
retrieval (hierarchical = two-stage document routing: route the query to the
best-matching topic folders via centroid embeddings, then rank chunks only
inside those topics), then runs the **constrained SLM** and the
**unconstrained baseline**, scoring grounding, hallucination rate, similarity,
latency, **peak GPU VRAM, and process RAM** per generation. Outputs:

- `outputs/results_per_question.csv` — every answer + every metric
- `outputs/summary.txt` — averaged tables + model memory footprint

Useful flags:
```bash
python scripts/3_run_experiments.py --tag qwen3b     # name this run
python scripts/3_run_experiments.py --adapter data/processed/lora_adapter
```
`--tag` keeps output files from different model-size runs separate (RQ3);
`--adapter` evaluates the fine-tuned condition (see Step 8).

Optional interactive mode:
```bash
python scripts/4_ask.py
```

---

## Step 8 (optional) — LoRA fine-tuning (RQ2 / RQ5)

Once your QA set has a decent number of items (20+ recommended; each needs a
`reference` answer), you can train a small LoRA adapter **using only module
data** — training examples are built from your QA references plus retrieved
module context, so no external data reaches the model:

```bash
pip install peft datasets
python scripts/5_finetune_lora.py            # builds data + trains 3 epochs
python scripts/3_run_experiments.py --adapter data/processed/lora_adapter
```

Compare `summary_lora.txt` against your prompt-only `summary.txt` to answer
"which constraint mechanism works best" (RQ2) with three conditions:
baseline (none) vs prompt-constrained RAG vs prompt-constrained RAG + adapter.
Training uses QLoRA (4-bit base + trainable rank-16 adapters) and fits on a
single T4. On CPU it is only practical as a smoke test.

---

## Choosing / changing the model

Edit `GEN_MODEL_NAME` in `src/config.py`. Defaults to
`Qwen/Qwen2.5-3B-Instruct` (no login token needed). For research question 3
(effect of model size), also try `Qwen/Qwen2.5-1.5B-Instruct` and
`Qwen/Qwen2.5-7B-Instruct`, rerun Step 7, and compare `summary.txt` files.

To use a Llama model you must `pip install huggingface_hub`, run
`huggingface-cli login`, and accept the license on its model page first.

---

## Troubleshooting (every error we anticipated)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran a script from inside `scripts/` | run from the **project root**: `python scripts/1_build_index.py` |
| `403 Forbidden` / can't reach huggingface.co | no internet / proxy | connect to the internet; corporate proxies block model downloads |
| `bitsandbytes` install fails | Windows/Mac/CPU | ignore — code falls back automatically |
| `CUDA out of memory` | model too big for GPU | use a smaller model (1.5B), keep 4-bit on, or lower `GEN_MAX_NEW_TOKENS` |
| `torch.cuda.is_available()` is False on a GPU box | CPU torch installed | reinstall torch with the CUDA index URL (Step 2) |
| Generation is very slow | running on CPU | expected; use a GPU, or a 1.5B model, or fewer QA items to test |
| `No documents found in data/raw` | folders not under `data/raw/` | check the folder actually contains `MOOC 1/...` etc. |
| Empty / weird PDF text | Google-Docs export spacing | already handled by `clean_text`; re-run build if you edited it |
| `precision@k = 0` everywhere | `gold_sources` don't match | copy exact `source` strings printed by `2_make_qa_set.py` |
| `ImportError: peft` | fine-tuning extras missing | `pip install peft datasets` (only needed for Step 8) |
| LoRA training OOM | context too long / rank too high | lower `max_length` in `finetune.py`, use `--rank 8`, or a 1.5B model |
| `ram_mb` is 0 on Windows | no psutil | `pip install psutil` |
| hierarchical = hybrid results | corpus has few topic folders | expected when routing can't exclude much; add more modules |
| old index errors after update | index built by previous version | delete `data/processed/` contents and re-run `1_build_index.py` |

---

## How the pieces map to the proposal

- **RQ1 (retrieve/ground/synthesise from closed corpus):** whole pipeline;
  grounding score.
- **RQ2 (which constraint prevents hallucination):** three conditions —
  baseline (no constraint) vs prompt-constrained RAG vs RAG + LoRA adapter
  (`--adapter`). Compare hallucination_rate across the summaries.
- **RQ3 (model size vs accuracy/latency/memory):** swap `GEN_MODEL_NAME`, run
  with different `--tag`s, compare latency_s / peak_vram_mb / ram_mb columns.
- **RQ4 (best retrieval pipeline):** dense vs lexical vs hybrid vs
  hierarchical routing, precision@k / recall@k.
- **RQ5 (prompt-tuning / adapter fine-tuning without external data):**
  constrained system prompt in `generator.py` + `scripts/5_finetune_lora.py`,
  which trains only on module-derived examples.
- **RQ6 (vs general LLM):** baseline rows are your general-LLM comparison.
