"""
core.py
-------
Core logic for the Medical Q&A chatbot:
  1. EntityRecognizer  - lightweight dictionary-based NER for diseases,
                          symptoms, and treatments.
  2. Retriever          - TF-IDF based semantic retrieval over the MedQuAD
                          question bank.

Both are deliberately dependency-light (no spaCy / transformers) so the app
starts instantly and runs on CPU with no GPU or large model downloads.
"""
import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Curated keyword lists for symptom / treatment recognition.
# These are common, general-purpose terms. The disease list is instead built
# dynamically from the dataset's `focus` column (5000+ real disease names),
# which gives far broader disease coverage than any hand-written list could.
# --------------------------------------------------------------------------
SYMPTOM_TERMS = [
    "pain", "fever", "cough", "fatigue", "nausea", "vomiting", "diarrhea",
    "constipation", "headache", "migraine", "dizziness", "rash", "itching",
    "swelling", "inflammation", "bleeding", "bruising", "numbness",
    "tingling", "weakness", "shortness of breath", "chest pain",
    "abdominal pain", "back pain", "joint pain", "muscle pain", "cramping",
    "chills", "sweating", "weight loss", "weight gain", "loss of appetite",
    "insomnia", "anxiety", "depression", "confusion", "memory loss",
    "seizure", "tremor", "blurred vision", "vision loss", "hearing loss",
    "sore throat", "runny nose", "congestion", "wheezing", "palpitations",
    "irregular heartbeat", "high blood pressure", "low blood pressure",
    "difficulty breathing", "difficulty swallowing", "jaundice", "anemia",
]

TREATMENT_TERMS = [
    "surgery", "chemotherapy", "radiation therapy", "radiotherapy",
    "antibiotics", "antiviral", "antifungal", "vaccine", "vaccination",
    "insulin", "dialysis", "physical therapy", "physiotherapy",
    "medication", "painkillers", "analgesics", "steroids",
    "corticosteroids", "immunotherapy", "transplant", "biopsy",
    "bisphosphonates", "calcitonin", "chemoprevention", "hormone therapy",
    "antidepressants", "anticoagulants", "blood thinners", "statins",
    "beta blockers", "ace inhibitors", "diuretics", "rehabilitation",
    "counseling", "psychotherapy", "occupational therapy", "screening",
    "monitoring", "lifestyle changes", "diet and exercise",
]


def _build_pattern(terms):
    """Compile a single regex that matches any of the given terms as whole words/phrases."""
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


@dataclass
class Entities:
    diseases: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    treatments: list = field(default_factory=list)

    def is_empty(self):
        return not (self.diseases or self.symptoms or self.treatments)


class EntityRecognizer:
    """Dictionary-based medical entity recognizer.

    Diseases are matched against the ~5,000 unique `focus` terms present in
    MedQuAD itself (real disease/condition names extracted from NIH/CDC/NLM
    sources). Symptoms and treatments are matched against curated keyword
    lists. Matching is case-insensitive and prefers longer phrase matches
    over shorter substrings.
    """

    def __init__(self, disease_names):
        # Keep only reasonably-sized disease names to avoid noisy 1-2 char matches
        cleaned = sorted({d.strip() for d in disease_names if d and len(d.strip()) > 2},
                          key=len, reverse=True)
        self._disease_names = cleaned
        # Build a lowercase lookup set for fast matching
        self._disease_lookup = {d.lower(): d for d in cleaned}
        self.symptom_pattern = _build_pattern(SYMPTOM_TERMS)
        self.treatment_pattern = _build_pattern(TREATMENT_TERMS)

    def _match_diseases(self, text_lower):
        found = []
        # Simple substring scan; disease list is pre-sorted longest-first so
        # multi-word disease names are preferred over short generic words.
        for name_lower, original in self._disease_lookup.items():
            if name_lower in text_lower:
                # avoid matching disease names that are themselves too generic (1 word, very short)
                found.append(original)
                if len(found) >= 5:  # cap to avoid runaway matches on very generic text
                    break
        return found

    def extract(self, text: str) -> Entities:
        text_lower = text.lower()
        symptoms = sorted({m.group(0).lower() for m in self.symptom_pattern.finditer(text)})
        treatments = sorted({m.group(0).lower() for m in self.treatment_pattern.finditer(text)})
        diseases = self._match_diseases(text_lower)
        # Drop disease matches that are just a symptom/treatment term wearing a
        # disease-name hat (e.g. "Chest Pain" is both a symptom and a MedlinePlus
        # health-topic focus name) to keep the categories clean and non-overlapping.
        claimed = set(symptoms) | set(treatments)
        diseases = [d for d in diseases if d.lower() not in claimed]
        return Entities(diseases=diseases, symptoms=symptoms, treatments=treatments)


class Retriever:
    """TF-IDF based retrieval over the MedQuAD question bank.

    We vectorize each `question` (optionally concatenated with its `focus`
    disease name for extra signal) and retrieve the closest matches to the
    user's query by cosine similarity.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        # Combine question + focus so retrieval also benefits from disease context
        corpus = (self.df["question"].fillna("") + " " + self.df["focus"].fillna(""))
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results[results["similarity"] > 0]  # drop zero-similarity noise


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["question", "answer"])
    return df
