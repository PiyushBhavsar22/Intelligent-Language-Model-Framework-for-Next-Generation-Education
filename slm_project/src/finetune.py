"""
finetune.py
===========
Lightweight adapter (LoRA) fine-tuning for RQ2 / RQ5.

RQ5 asks: "Can prompt-tuning or adapter-based fine-tuning encode the expected
structure, terminology, and reasoning style WITHOUT exposing the model to
external data?"

So the training data here comes ONLY from the module corpus itself:
  * build_training_data() creates instruction-style examples where the input
    is (retrieved context + question) and the target is a grounded answer
    drawn from your QA set (data/qa/qa.jsonl). No external data is used.
  * train_lora() attaches small LoRA adapters (rank-r matrices on the
    attention projections) and trains only those — the base model stays
    frozen, so this runs on a single T4 with 4-bit quantization (QLoRA).

Outputs go to data/processed/lora_adapter/ and can be loaded by
Generator(adapter_path=...).

Extra dependencies (add to your environment):
    pip install peft datasets
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from src.config import (
    QA_DIR, PROCESSED_DIR, GEN_MODEL_NAME, TOP_K,
)

TRAIN_FILE   = PROCESSED_DIR / "lora_train.jsonl"
ADAPTER_DIR  = PROCESSED_DIR / "lora_adapter"

# Same constrained instruction the inference prompt uses, so the adapter
# learns the exact behaviour we evaluate.
INSTRUCTION = (
    "You are a teaching assistant for a single university module. "
    "Answer the student's question USING ONLY the numbered context passages "
    "provided. Do not use any outside knowledge. If the answer is not "
    "contained in the context, reply exactly: "
    "\"I cannot answer this based on the module materials.\""
)


# ----------------------------------------------------------------------
# 1) Build training data from the QA set + retriever (module data only)
# ----------------------------------------------------------------------
def build_training_data(qa_file: Path | None = None) -> int:
    """
    For each QA item, retrieve context with the hybrid retriever and write a
    (prompt, target) pair. Returns number of examples written.

    Uses the *reference* answers you wrote in qa.jsonl as targets — i.e. the
    adapter learns the module's terminology and answer style from module
    material alone.
    """
    from src.retriever import Retriever  # imported here to avoid heavy deps at module import

    qa_file = qa_file or (QA_DIR / "qa.jsonl")
    if not qa_file.exists():
        raise FileNotFoundError(
            f"{qa_file} not found. Create it with scripts/2_make_qa_set.py "
            "and add your tutorial/exam questions first."
        )
    items = [json.loads(l) for l in open(qa_file, encoding="utf-8")]
    retriever = Retriever()

    n = 0
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for it in items:
            q, ref = it["question"], it.get("reference", "").strip()
            if not ref:
                continue
            ctx = retriever.search(q, top_k=TOP_K, method="hybrid")
            ctx_block = "\n\n".join(
                f"[{i}] {c['text']}" for i, c in enumerate(ctx, 1)
            )
            user = f"Context passages:\n{ctx_block}\n\nQuestion: {q}\n\nAnswer:"
            f.write(json.dumps(
                {"system": INSTRUCTION, "user": user, "target": ref},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} training examples to {TRAIN_FILE}")
    if n < 20:
        print("[warn] fewer than 20 examples — LoRA results will be weak. "
              "Add more QA pairs to data/qa/qa.jsonl.")
    return n


# ----------------------------------------------------------------------
# 2) Train the LoRA adapter (QLoRA when a GPU is present)
# ----------------------------------------------------------------------
def train_lora(epochs: int = 3, lr: float = 2e-4, rank: int = 16,
               model_name: str = GEN_MODEL_NAME) -> Path:
    """
    Train LoRA adapters on the examples in lora_train.jsonl.
    Saves the adapter to data/processed/lora_adapter/ and returns that path.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        raise ImportError(
            "Fine-tuning needs extra packages: pip install peft datasets"
        ) from e

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "No training data. Run build_training_data() first "
            "(see scripts/5_finetune_lora.py)."
        )

    device_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- load base model (4-bit on GPU = QLoRA; fp32 on CPU for smoke tests) ----
    load_kwargs = {}
    if device_cuda:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto" if device_cuda else None, **load_kwargs
    )
    if device_cuda and "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- tokenise: full chat-templated text, loss over the whole sequence ----
    rows = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8")]

    def to_text(r):
        msgs = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["target"]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    ds = Dataset.from_dict({"text": [to_text(r) for r in rows]})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1536)

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(PROCESSED_DIR / "lora_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=device_cuda,
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=collator).train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")
    return ADAPTER_DIR
