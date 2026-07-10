"""
generator.py
============
Loads a small open-source instruct model and generates answers.

Two modes (central to the research questions):
  * generate_constrained(question, contexts)
        Strict RAG. Model may ONLY use the retrieved chunks. If the answer is
        not in the context it must say so. This is the "constrained SLM".
  * generate_baseline(question)
        No context, no constraint — the model answers from its parametric
        knowledge. Used to compare hallucination rate.

Robustness:
  * Auto-detects GPU. Uses 4-bit quantization only if bitsandbytes + CUDA are
    both available; otherwise cleanly falls back to fp16/fp32 on whatever device
    exists, so the SAME code runs on a laptop CPU or a T4 GPU.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    GEN_MODEL_NAME, GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, LOAD_IN_4BIT,
)

# ---------- prompt templates ----------
CONSTRAINED_SYSTEM = (
    "You are a teaching assistant for a single university module. "
    "Answer the student's question USING ONLY the numbered context passages provided. "
    "Do not use any outside knowledge. "
    "If the answer is not contained in the context, reply exactly: "
    "\"I cannot answer this based on the module materials.\" "
    "Be concise and factual. When you use a passage, you may cite it like [1], [2]."
)

BASELINE_SYSTEM = (
    "You are a helpful assistant. Answer the question concisely and factually."
)


def _build_context_block(contexts: list[dict]) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] (source: {c.get('source', '?')})\n{c['text']}")
    return "\n\n".join(lines)


class Generator:
    def __init__(self, model_name: str = GEN_MODEL_NAME,
                 adapter_path: str | None = None) -> None:
        """
        adapter_path: optional path to a LoRA adapter directory produced by
        src/finetune.py (e.g. data/processed/lora_adapter). When given, the
        adapter is attached on top of the frozen base model — this is the
        'fine-tuned constrained SLM' condition for RQ2/RQ5.
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {}
        quantized = False
        if LOAD_IN_4BIT and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401  (import test)
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                quantized = True
            except Exception as e:
                print(f"[info] 4-bit unavailable ({e}); loading in fp16.")

        if not quantized:
            load_kwargs["torch_dtype"] = (
                torch.float16 if self.device == "cuda" else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto" if self.device == "cuda" else None,
            **load_kwargs,
        )
        if self.device == "cpu":
            self.model.to("cpu")

        # ---- optional LoRA adapter (fine-tuned condition, RQ2/RQ5) ----
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "Loading a LoRA adapter requires peft: pip install peft"
                ) from e
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
            print(f"Attached LoRA adapter from {adapter_path}")

        self.model.eval()
        print(f"Loaded {model_name} on {self.device} "
              f"({'4-bit' if quantized else 'full/half precision'}"
              f"{', +LoRA' if adapter_path else ''}).")

    # ---------- core generation ----------
    @torch.no_grad()
    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = GEN_TEMPERATURE > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=GEN_TEMPERATURE if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ---------- public API ----------
    def generate_constrained(self, question: str, contexts: list[dict]) -> str:
        ctx = _build_context_block(contexts)
        user = f"Context passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
        return self._chat(CONSTRAINED_SYSTEM, user)

    def generate_baseline(self, question: str) -> str:
        return self._chat(BASELINE_SYSTEM, question)
