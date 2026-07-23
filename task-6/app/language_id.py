"""
Language identification for the multilingual chatbot.

Uses `langdetect` (a pure-Python port of Google's language-detection
library) -- fully offline, no model download required, unlike neural
language-ID models that need weights pulled from a model hub.

Supports: English (en), Spanish (es), French (fr), Hindi (hi) -- the base
language plus the three additional languages required by the task.
"""

import re
from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # deterministic results

SUPPORTED_LANGUAGES = {"en", "es", "fr", "hi"}
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi"}


def detect_language(text: str) -> dict:
    """Detects the language of a full message, restricted to supported
    languages. Returns the best supported guess with a confidence score,
    even if langdetect's top guess is an unsupported language (falls back
    to the highest-confidence supported language in its ranked list)."""
    text = text.strip()
    if not text:
        return {"lang": "en", "confidence": 0.0, "supported": True}

    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return {"lang": "en", "confidence": 0.0, "supported": False}

    for c in candidates:
        if c.lang in SUPPORTED_LANGUAGES:
            return {"lang": c.lang, "confidence": round(c.prob, 3), "supported": True}

    # top guess isn't one of our 4 supported languages
    top = candidates[0]
    return {"lang": top.lang, "confidence": round(top.prob, 3), "supported": False}


def split_into_chunks(text: str):
    """Splits a message on sentence-ish boundaries so mixed-language
    messages (e.g. half English, half Spanish in one message) can be
    detected per-chunk instead of forcing one language label onto the
    whole message."""
    chunks = re.split(r"(?<=[.!?।])\s+|(?<=[,;])\s+(?=[A-Za-zÀ-ÿऀ-ॿ])", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def detect_mixed_language(text: str) -> dict:
    """Detects language per chunk of a message. Returns the dominant
    language (by chunk count) plus a per-chunk breakdown, so the app can
    tell the user "this looks like a mix of X and Y" when relevant."""
    chunks = split_into_chunks(text)
    if len(chunks) <= 1:
        d = detect_language(text)
        return {"dominant": d["lang"], "is_mixed": False, "chunks": [{"text": text, **d}]}

    chunk_results = []
    for c in chunks:
        d = detect_language(c)
        chunk_results.append({"text": c, **d})

    langs_seen = {c["lang"] for c in chunk_results if c["supported"]}
    counts = {}
    for c in chunk_results:
        counts[c["lang"]] = counts.get(c["lang"], 0) + 1
    dominant = max(counts, key=counts.get)

    return {
        "dominant": dominant,
        "is_mixed": len(langs_seen) > 1,
        "chunks": chunk_results,
    }
