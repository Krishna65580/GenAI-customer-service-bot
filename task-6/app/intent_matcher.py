"""
Cross-lingual intent matching.

Rather than only matching a message's keywords against the keyword list of
its *detected* language, this searches ALL supported languages' keyword
lists against the message. This is what actually makes mixed-language /
code-switched input work: a message that's half English, half Spanish
("Hola, where is my order?") still matches the order_status intent even
though no single language's keyword list alone would cover the sentence.

This is the "cross-lingual reasoning" piece: intent understanding is
decoupled from which language the words happen to be in.
"""

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def score_intents(text: str, kb: dict) -> dict:
    """Returns {intent_name: score} where score = sum of word-counts of
    every matched keyword phrase across ALL languages' keyword lists
    (case-insensitive substring match).

    Weighting by phrase length (not just hit count) matters for mixed-
    language messages: e.g. "Hola, where is my order?" should match
    order_status, not greeting, even though both "hola" (1 word) and
    "where is my order" (4 words) are technically substring hits -- a
    flat +1-per-hit score would tie them and break the tie arbitrarily.
    """
    t = _normalize(text)
    scores = {}
    for intent_name, intent_data in kb["intents"].items():
        weight = 0
        for lang, keywords in intent_data["keywords"].items():
            for kw in keywords:
                kw_norm = _normalize(kw)
                if kw_norm in t:
                    weight += len(kw_norm.split())
        if weight > 0:
            scores[intent_name] = weight
    return scores


def best_intent(text: str, kb: dict, min_score: int = 1):
    scores = score_intents(text, kb)
    if not scores:
        return None
    top = max(scores, key=scores.get)
    if scores[top] < min_score:
        return None
    return top


def get_response(intent_name: str, lang: str, kb: dict) -> str:
    intent = kb["intents"].get(intent_name)
    if not intent:
        return kb["fallback"].get(lang, kb["fallback"]["en"])
    responses = intent["response"]
    return responses.get(lang, responses["en"])
