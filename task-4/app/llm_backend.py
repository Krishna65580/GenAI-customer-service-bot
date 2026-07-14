"""
Pluggable open-source LLM backend.

The assignment asks for "user open source LLM for explanation generation".
This module wires the app up to a locally running open-source LLM served by
Ollama (https://ollama.com) — e.g. Llama 3, Mistral, Phi-3 — over its local
HTTP API. No API key or cloud service required, which satisfies "open source
LLM" in the literal sense (weights + inference run locally).

If Ollama is not installed/running (as in this sandboxed build environment,
which has no access to model-weight hosts), `get_llm_backend()` returns None
and the app automatically falls back to the deterministic extractive/
template-based explanation generator in nlp_engine.py, so the app is always
runnable end-to-end.

To enable a real LLM on your own machine:
    1. Install Ollama: https://ollama.com/download
    2. Run:  ollama pull llama3.2        (or mistral, phi3, qwen2.5, etc.)
    3. Run:  ollama serve                (usually auto-started)
    4. Launch the Streamlit app - it will auto-detect Ollama and use it.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.RequestException:
        return ""


def get_llm_backend(model: str = DEFAULT_MODEL):
    """Returns a callable(prompt) -> str if a local open-source LLM is
    reachable, otherwise None."""
    if _ollama_available():
        return lambda prompt: _call_ollama(prompt, model=model)
    return None
