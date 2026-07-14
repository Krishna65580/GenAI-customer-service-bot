"""
Core NLP engine for the arXiv Domain-Expert Chatbot.

Implements, without any external model downloads (the sandbox this was built in
has no internet access to Kaggle / HuggingFace / arXiv's API), a fully working
NLP pipeline:

  1. Information Retrieval  -> TF-IDF vector space model + cosine similarity
  2. Summarization           -> extractive, sentence-graph (TextRank-style)
  3. Keyword extraction      -> TF-IDF top-terms per document
  4. Explanation generation  -> retrieval-grounded template composition
                                 (pluggable: swap in a real open-source LLM,
                                 see llm_backend.py, with zero other changes)

This keeps the whole app runnable offline / in restricted environments while
being architected so a real LLM (Llama / Mistral via Ollama, or a HF pipeline)
can be dropped into `generate_explanation()` unchanged from the outside.
"""

import re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_papers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["content"] = (
        df["title"].fillna("") + ". " +
        df["summary"].fillna("") + " Keywords: " +
        df["keywords"].fillna("")
    )
    return df


class ArxivSearchEngine:
    """TF-IDF based semantic-ish search over the paper corpus."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.df["content"])

    def search(self, query: str, top_k: int = 5, category: str = None) -> pd.DataFrame:
        sub_df = self.df
        sub_matrix = self.doc_matrix
        if category and category != "All":
            mask = self.df["category"] == category
            sub_df = self.df[mask].reset_index(drop=True)
            sub_matrix = self.doc_matrix[mask.values]
            if sub_df.empty:
                return sub_df.assign(score=[])

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, sub_matrix).flatten()
        order = np.argsort(-sims)[:top_k]
        result = sub_df.iloc[order].copy()
        result["score"] = sims[order]
        return result[result["score"] > 0.0]

    def top_keywords(self, row_idx: int, n: int = 8):
        """TF-IDF-ranked keywords for a single document."""
        vec = self.doc_matrix[row_idx].toarray().flatten()
        terms = np.array(self.vectorizer.get_feature_names_out())
        top_idx = np.argsort(-vec)[:n]
        return [t for t in terms[top_idx] if vec[top_idx[list(terms[top_idx]).index(t)]] > 0]


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if len(s.split()) > 3]


def extractive_summarize(text: str, n_sentences: int = 2) -> str:
    """TextRank-style extractive summarizer: build a sentence similarity graph
    from TF-IDF vectors, rank sentences with PageRank, return the top-N in
    original order."""
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:n_sentences])

    sim_matrix = cosine_similarity(matrix)
    np.fill_diagonal(sim_matrix, 0)
    graph = nx.from_numpy_array(sim_matrix)

    try:
        scores = nx.pagerank(graph, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        scores = {i: 1.0 for i in range(len(sentences))}

    ranked = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
    chosen = sorted(i for _, i in ranked[:n_sentences])
    return " ".join(sentences[i] for i in chosen)


def multi_doc_summary(papers: pd.DataFrame, n_sentences_each: int = 1) -> str:
    parts = []
    for _, row in papers.iterrows():
        parts.append(f"- **{row['title']}** ({row['year']}): {row['summary']}")
    return "\n".join(parts)


def generate_explanation(query: str, retrieved: pd.DataFrame, llm_backend=None) -> str:
    """Retrieval-grounded explanation generation.

    If an `llm_backend` callable is supplied (e.g. a wrapper around a local
    open-source LLM through Ollama, see llm_backend.py), it is used to
    synthesize a fluent answer conditioned on the retrieved context
    (a small Retrieval-Augmented-Generation loop). Otherwise a deterministic,
    template-based composition of the retrieved evidence is returned, so the
    app is always fully functional offline.
    """
    if retrieved.empty:
        return (
            "I couldn't find a paper in this knowledge base closely related to "
            "that question. Try rephrasing, or ask about a core topic such as "
            "transformers, diffusion models, reinforcement learning, or word "
            "embeddings."
        )

    context = "\n\n".join(
        f"Paper: {r['title']} ({r['year']})\nSummary: {r['summary']}"
        for _, r in retrieved.iterrows()
    )

    if llm_backend is not None:
        prompt = (
            "You are a domain-expert research assistant for computer science / "
            "AI. Using ONLY the context below, answer the user's question "
            "clearly, and mention which paper(s) support each claim.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        answer = llm_backend(prompt)
        if answer:
            return answer

    # --- Deterministic fallback (no external LLM required) ---
    top = retrieved.iloc[0]
    lines = [
        f"Based on the closest matching paper, **{top['title']}** ({top['year']}) "
        f"by {top['authors']}:",
        "",
        top["summary"],
    ]
    if len(retrieved) > 1:
        lines.append("")
        lines.append("Related papers that also touch on this:")
        for _, r in retrieved.iloc[1:].iterrows():
            lines.append(f"- {r['title']} ({r['year']}) — {r['summary'][:140]}...")
    return "\n".join(lines)
