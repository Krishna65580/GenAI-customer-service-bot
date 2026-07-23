"""
Core retrieval + basic medical NER for the MedQuAD Q&A chatbot.

Retrieval: TF-IDF vector space model + cosine similarity over the real
MedQuAD question/answer pairs (data/medquad.csv, parsed from
https://github.com/abachaa/MedQuAD by build_dataset.py).

Entity recognition: lexicon-based matching against curated lists of common
symptoms, diseases, and treatments (app/medical_lexicon.json). This is a
transparent, inspectable approach appropriate for a demo; a production
system would use a trained clinical NER model (e.g. scispaCy, MedCAT) --
see README for how to swap it in.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "medquad.csv")
LEXICON_PATH = os.path.join(BASE_DIR, "medical_lexicon.json")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["search_text"] = df["focus"].fillna("") + ". " + df["question"].fillna("")
    return df


def load_lexicon() -> dict:
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class MedQARetriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def search(self, query: str, top_k: int = 5, qtype: str = None):
        sub_df = self.df
        sub_matrix = self.matrix
        if qtype and qtype != "All":
            mask = self.df["qtype"] == qtype
            sub_df = self.df[mask].reset_index(drop=True)
            sub_matrix = self.matrix[mask.values]
            if sub_df.empty:
                return sub_df.assign(score=[])

        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, sub_matrix).flatten()
        order = np.argsort(-sims)[:top_k]
        result = sub_df.iloc[order].copy()
        result["score"] = sims[order]
        return result[result["score"] > 0.0]


class MedicalEntityRecognizer:
    """Lexicon-based entity matcher for symptoms / diseases / treatments."""

    def __init__(self, lexicon: dict):
        self.lexicon = lexicon
        self._patterns = {
            category: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                       for term in terms]
            for category, terms in lexicon.items()
        }

    def extract(self, text: str) -> dict:
        found = {"symptoms": [], "diseases": [], "treatments": []}
        for category, patterns in self._patterns.items():
            for term, pat in zip(self.lexicon[category], patterns):
                if pat.search(text):
                    found[category].append(term)
        return found

    def has_entities(self, entities: dict) -> bool:
        return any(len(v) > 0 for v in entities.values())


def compose_answer(query: str, results: pd.DataFrame, entities: dict) -> str:
    if results.empty:
        return (
            "I couldn't find a close match for that in the MedQuAD knowledge base. "
            "Try rephrasing, or ask about a specific condition, its symptoms, causes, "
            "or treatment (e.g. \"What are the symptoms of asthma?\")."
        )

    top = results.iloc[0]
    lines = [f"**{top['focus']}** — _{top['question']}_", "", top["answer"][:1200]]
    if len(top["answer"]) > 1200:
        lines.append("...(truncated)")

    entity_notes = []
    for cat, items in entities.items():
        if items:
            entity_notes.append(f"{cat}: {', '.join(sorted(set(items)))}")
    if entity_notes:
        lines.append("")
        lines.append("🏷️ Detected medical terms in your question — " + " | ".join(entity_notes))

    if len(results) > 1:
        lines.append("")
        lines.append("Related questions in the knowledge base:")
        for _, r in results.iloc[1:].iterrows():
            lines.append(f"- {r['question']} _(source: {r['source']})_")

    return "\n".join(lines)
