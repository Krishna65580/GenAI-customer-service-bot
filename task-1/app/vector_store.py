"""
Dynamically-expanding knowledge base for a RAG chatbot.

Uses TF-IDF vectors as the embedding space and cosine similarity for
retrieval (a lightweight, dependency-free stand-in for a real vector
database like Chroma/FAISS/Pinecone -- swappable, see the README). The
important part for this task is the *mechanism*, not the embedding
algorithm: documents can be added at any time (manually, via file upload,
or pulled automatically from a queue of "external sources"), the index is
rebuilt incrementally, and every document is timestamped so the chatbot's
knowledge base grows over time without needing to be redeployed.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.json")
QUEUE_PATH = os.path.join(BASE_DIR, "data", "incoming_sources.json")
LOG_PATH = os.path.join(BASE_DIR, "data", "ingestion_log.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class VectorKnowledgeBase:
    def __init__(self):
        self.docs = _load_json(KB_PATH, [])
        self.log = _load_json(LOG_PATH, [])
        self._build_index()

    # ---------- indexing ----------
    def _build_index(self):
        if not self.docs:
            self.vectorizer = None
            self.matrix = None
            return
        corpus = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def persist(self):
        _save_json(KB_PATH, self.docs)
        _save_json(LOG_PATH, self.log)

    # ---------- mutation ----------
    def add_document(self, title: str, text: str, source: str = "manual"):
        new_id = (max((d["id"] for d in self.docs), default=0)) + 1
        doc = {
            "id": new_id,
            "source": source,
            "title": title.strip(),
            "text": text.strip(),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.docs.append(doc)
        self.log.append({
            "event": "add",
            "doc_id": new_id,
            "title": doc["title"],
            "source": source,
            "timestamp": doc["added_at"],
        })
        self._build_index()
        self.persist()
        return doc

    # ---------- periodic ingestion ----------
    def ingest_next_from_queue(self):
        """Pulls the next pending item from the simulated external-source
        queue and adds it to the knowledge base. In production, replace this
        with a real fetch (API poll, RSS feed, database delta, S3 bucket
        listing, etc.) -- the ingestion mechanism and KB update logic stay
        identical."""
        queue = _load_json(QUEUE_PATH, [])
        if not queue:
            return None
        item = queue.pop(0)
        _save_json(QUEUE_PATH, queue)
        return self.add_document(item["title"], item["text"], source=item["source"])

    def pending_count(self):
        return len(_load_json(QUEUE_PATH, []))

    # ---------- retrieval ----------
    def search(self, query: str, top_k: int = 3):
        if not self.docs or self.vectorizer is None:
            return []
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix).flatten()
        order = np.argsort(-sims)[:top_k]
        results = []
        for i in order:
            if sims[i] > 0:
                d = dict(self.docs[i])
                d["score"] = float(sims[i])
                results.append(d)
        return results

    def stats(self):
        last_update = max((d["added_at"] for d in self.docs), default=None)
        by_source = {}
        for d in self.docs:
            by_source[d["source"]] = by_source.get(d["source"], 0) + 1
        return {
            "total_docs": len(self.docs),
            "last_update": last_update,
            "by_source": by_source,
        }
