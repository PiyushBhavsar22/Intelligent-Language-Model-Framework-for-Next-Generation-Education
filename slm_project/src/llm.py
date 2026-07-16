from __future__ import annotations
import logging
import time
from dataclasses import dataclass
import requests
from config import CONFIG, Config

log = logging.getLogger("llm")

class OllamaError(RuntimeError):
    pass

@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    eval_tokens: int
    tokens_per_s: float
    prompt_tokens: int

class OllamaClient:
    def __init__(self, cfg: Config = CONFIG, model: str | None = None):
        self.cfg = cfg
        self.model = model or cfg.model_name
        self.base = cfg.ollama_url.rstrip("/")

    # health
    def ping(self) -> bool:
        try:
            return requests.get(f"{self.base}/api/tags",
                                timeout=5).status_code == 200
        except requests.ConnectionError:
            return False

    def available_models(self) -> list[str]:
        r = requests.get(f"{self.base}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def ensure_ready(self) -> None:
        if not self.ping():
            raise OllamaError(
                f"Cannot reach Ollama at {self.base}. Start the daemon "
                f"(`ollama serve` or the Ollama app) and retry.")
        models = self.available_models()
        if not any(m == self.model or m.split(":")[0] == self.model
                   for m in models):
            raise OllamaError(
                f"Model '{self.model}' is not pulled. Fix with:\n"
                f"    ollama pull {self.model}\n"
                f"Available: {models or 'none'}")

    # generation
    def generate(self, system: str, user: str,
                 temperature: float | None = None,
                 max_new_tokens: int | None = None) -> LLMResult:
        if not user.strip():
            raise ValueError("Empty prompt - refusing to call Ollama "
                             "(check retrieval output upstream).")
        payload = {
            "model": self.model, "system": system, "prompt": user,
            "stream": False, "keep_alive": "24h", "think": False,
            "options": {
                "temperature": (self.cfg.temperature if temperature is None
                                else temperature),
                "top_p": self.cfg.top_p,
                "top_k": self.cfg.top_k_sampling,
                "num_predict": max_new_tokens or self.cfg.max_new_tokens,
                "num_ctx": self.cfg.num_ctx,
                "seed": self.cfg.seed,
            },
        }
        fallbacks = list(self.cfg.fallback_models)
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.request_retries + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{self.base}/api/generate", json=payload,
                                  timeout=self.cfg.request_timeout_s)
                if r.status_code == 404:
                    raise OllamaError(
                        f"Model '{payload['model']}' not found. Run: "
                        f"ollama pull {payload['model']}")
                if r.status_code == 400 and "think" in r.text.lower():
                    payload.pop("think", None)
                    continue          # retry immediately without the field
                if r.status_code == 500:
                    low = r.text.lower()
                    crash_signals = ("memory", "cuda", "model runner",
                                     "unexpectedly stopped",
                                     "resource limitation")
                    if any(sig in low for sig in crash_signals):
                        raise MemoryError(r.text[:300])
                    raise OllamaError(
                        f"Ollama returned 500 for model "
                        f"'{payload['model']}': {r.text[:300]}")
                r.raise_for_status()
                d = r.json()
                latency = time.perf_counter() - t0
                text = d.get("response", "").strip()
                if not text and d.get("thinking"):
                    # Thinking model burned the budget on reasoning; salvage
                    # nothing — treat as retryable, the retry will have
                    # think=False honored or a bigger budget.
                    log.warning("Model produced only 'thinking' output "
                                "(%d chars), no response.",
                                len(d.get("thinking", "")))
                prompt_tokens = int(d.get("prompt_eval_count", 0))
                n = int(d.get("eval_count", 0))
                dur = max(int(d.get("eval_duration", 0)) / 1e9, 1e-9)

                if not text:
                    done_reason = d.get("done_reason", "unknown")
                    log.warning(
                        "Empty response from %s: done_reason=%s, "
                        "prompt_tokens=%d, num_ctx=%d, eval_count=%d. "
                        "Likely context overflow or prompt-triggered refusal.",
                        payload["model"], done_reason, prompt_tokens,
                        self.cfg.num_ctx, n)
                    if prompt_tokens >= self.cfg.num_ctx - 128:
                        raise OllamaError(
                            f"Prompt ({prompt_tokens} tok) exceeds num_ctx "
                            f"({self.cfg.num_ctx}). Increase SLMQA_NUM_CTX.")
                    raise OllamaError(
                        f"Empty response (done_reason={done_reason})")

                return LLMResult(text, payload["model"], latency, n, n / dur,
                                 prompt_tokens)
            except requests.ConnectionError as exc:
                raise OllamaError(
                    f"Connection refused at {self.base} - Ollama is not "
                    f"running (`ollama serve`).") from exc
            except MemoryError as exc:
                if fallbacks:
                    nxt = fallbacks.pop(0)
                    log.warning("VRAM OOM on %s -> falling back to %s",
                                payload["model"], nxt)
                    payload["model"] = self.model = nxt
                    continue
                raise OllamaError(
                    "OOM on every model in the fallback chain. Lower num_ctx "
                    "or close other GPU applications.") from exc
            except (requests.Timeout, requests.HTTPError, OllamaError) as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning("Ollama attempt %d/%d failed (%s); retry in %ds",
                            attempt, self.cfg.request_retries, exc, wait)
                time.sleep(wait)
        raise OllamaError(f"Ollama failed after {self.cfg.request_retries} "
                          f"retries: {last_exc}")


def gpu_memory_mb() -> float | None:
    """Whole-GPU used memory via nvidia-smi (Ollama lives outside torch)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None
