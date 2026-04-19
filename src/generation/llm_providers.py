"""
llm_providers.py - Unified make_llm(name) -> callable(prompt: str) -> str.

Supported:
  - "gemini-2.5-flash", "gemini-2.0-flash", "gemini-*"   (Google Gemini API)
  - "vllm:<served-model-name>"                           (vLLM OpenAI-compat server)
  - "Qwen/Qwen2.5-1.5B-Instruct" or any HF text-gen repo (local transformers)

Gemini auth:  GEMINI_API_KEY (or GOOGLE_API_KEY).
vLLM config:  VLLM_BASE_URL (default http://localhost:8000/v1), VLLM_API_KEY (default EMPTY).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable


def _load_dotenv_if_present() -> None:
    """Load KEY=value lines from a project-root .env into os.environ (idempotent)."""
    for env_path in [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
        return


def _make_gemini(model: str, **kwargs) -> Callable[[str], str]:
    from google import genai
    from google.genai import types

    _load_dotenv_if_present()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) env var to use Gemini."
        )
    client = genai.Client(api_key=api_key)

    cfg = types.GenerateContentConfig(
        temperature=kwargs.get("temperature", 0.0),
        max_output_tokens=kwargs.get("max_output_tokens", 1024),
    )

    def call(prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=cfg,
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after 3 retries: {last_err}")

    return call


def _make_vllm(model: str, **kwargs) -> Callable[[str], str]:
    """Talk to a vLLM server via its OpenAI-compatible /v1/chat/completions endpoint."""
    from openai import OpenAI

    _load_dotenv_if_present()
    base_url = kwargs.get("base_url") or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key  = kwargs.get("api_key")  or os.environ.get("VLLM_API_KEY", "EMPTY")
    client   = OpenAI(base_url=base_url, api_key=api_key)

    temperature = float(kwargs.get("temperature", 0.0))
    max_tokens  = int(kwargs.get("max_tokens", 2048))
    timeout     = float(kwargs.get("timeout", 120))

    def call(prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"vLLM call failed after 3 retries: {last_err}")

    return call


def _make_hf(model: str, **kwargs) -> Callable[[str], str]:
    """Local HuggingFace text-gen.  `quant` ∈ {"none","4bit","8bit"} (default: "none")."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    quant = kwargs.get("quant", "none").lower()
    has_gpu = torch.cuda.is_available()
    max_new = kwargs.get("max_new_tokens", 512)

    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        tok   = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        mdl   = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True,
        )
        gen = hf_pipeline("text-generation", model=mdl, tokenizer=tok,
                          max_new_tokens=max_new, do_sample=False)
    else:
        gen = hf_pipeline(
            "text-generation",
            model=model,
            device=0 if has_gpu else -1,
            torch_dtype=torch.float16 if has_gpu else None,
            max_new_tokens=max_new, do_sample=False,
            trust_remote_code=True,
        )

    def call(prompt: str) -> str:
        out = gen(prompt)[0]["generated_text"]
        return out[len(prompt):] if out.startswith(prompt) else out

    return call


def make_llm(name: str, **kwargs) -> Callable[[str], str]:
    low = name.lower()
    if low.startswith("gemini"):
        return _make_gemini(name, **kwargs)
    if low.startswith("vllm:"):
        return _make_vllm(name.split(":", 1)[1], **kwargs)
    return _make_hf(name, **kwargs)
