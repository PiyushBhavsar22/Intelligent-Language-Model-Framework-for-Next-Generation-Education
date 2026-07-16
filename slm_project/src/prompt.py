"""prompt.py - strict / normal / baseline """
from __future__ import annotations

from config import CONFIG, Config

STRICT_SYSTEM = """\
You are a teaching assistant for a single university module. You must answer
using ONLY the module material provided in the CONTEXT below.

Rules you must never break:
1. Use only facts stated in the CONTEXT. Do not use any outside knowledge.
2. After every claim, cite the supporting source tag, e.g. [lecture3.pdf, p.12].
3. If the CONTEXT does not contain the information needed, reply exactly:
   "{refusal}"
4. Never guess, never speculate, never fill gaps with general knowledge.
5. Be concise and use the module's own terminology."""

NORMAL_SYSTEM = """\
You are a helpful teaching assistant for a university module. Use the CONTEXT
below to inform your answer where relevant, citing sources like
[lecture3.pdf, p.12]."""

BASELINE_SYSTEM = """\
You are a helpful teaching assistant. Answer the student's question as well
as you can."""

USER_TEMPLATE = "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
BASELINE_USER = "QUESTION: {question}\n\nANSWER:"

CORRECTIVE_SUFFIX = (
    "\n\nIMPORTANT: your previous answer contained statements not supported "
    "by the CONTEXT. Rewrite it using ONLY facts that appear verbatim or "
    "near-verbatim in the CONTEXT, or refuse if the answer is not there.")


def build_prompt(question: str, context: str, mode: str = "strict",
                 corrective: bool = False,
                 cfg: Config = CONFIG) -> tuple[str, str]:
    if mode == "strict":
        sys = STRICT_SYSTEM.format(refusal=cfg.refusal_text)
        if corrective:
            sys += CORRECTIVE_SUFFIX
        return sys, USER_TEMPLATE.format(context=context, question=question)
    if mode == "normal":
        return NORMAL_SYSTEM, USER_TEMPLATE.format(context=context,
                                                   question=question)
    if mode == "baseline":
        return BASELINE_SYSTEM, BASELINE_USER.format(question=question)
    raise ValueError(f"Unknown prompt mode {mode!r} "
                     "(expected strict | normal | baseline)")


def is_refusal(answer: str, cfg: Config = CONFIG) -> bool:
    a = answer.strip().lower()
    return (cfg.refusal_text.lower() in a
            or "not covered in the module" in a
            or a.startswith("i cannot answer"))
