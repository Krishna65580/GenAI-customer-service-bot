import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nlp_engine import (
    load_papers, ArxivSearchEngine, extractive_summarize,
    multi_doc_summary, generate_explanation
)
from llm_backend import get_llm_backend

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "papers.csv")

st.set_page_config(page_title="arXiv CS/AI Expert Chatbot", page_icon="🧠", layout="wide")


@st.cache_resource
def load_engine():
    df = load_papers(DATA_PATH)
    engine = ArxivSearchEngine(df)
    return df, engine


@st.cache_resource
def load_llm():
    return get_llm_backend()


df, engine = load_engine()
llm_backend = load_llm()

# ---------------- Sidebar ----------------
st.sidebar.title("🧠 arXiv Domain Expert")
st.sidebar.caption("Computer Science / AI subset")
page = st.sidebar.radio("Navigate", ["💬 Chat", "🔍 Paper Search", "📊 Concept Visualization", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Papers indexed:** {len(df)}")
st.sidebar.markdown(f"**Categories:** {', '.join(sorted(df['category'].unique()))}")
if llm_backend:
    st.sidebar.success("Open-source LLM (Ollama) connected ✅")
else:
    st.sidebar.info("Running in offline mode — extractive/template NLP "
                     "(no local LLM detected). See About tab to enable one.")

# ---------------- Chat Page ----------------
if page == "💬 Chat":
    st.title("💬 Ask the arXiv Domain Expert")
    st.caption("Ask about transformers, diffusion models, RL, embeddings, vision models, and more. "
               "Answers are grounded in the paper knowledge base (RAG-style retrieval).")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm an AI/CS research assistant. Ask me to explain a "
                                              "concept, summarize a paper, or compare two techniques — "
                                              "e.g. *\"How does attention work in transformers?\"* or "
                                              "*\"Compare GANs and diffusion models.\"*"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about CS / AI research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant papers and composing answer..."):
                category = None
                results = engine.search(prompt, top_k=3)
                answer = generate_explanation(prompt, results, llm_backend=llm_backend)

                if not results.empty:
                    sources = ", ".join(
                        f"*{r['title']}* ({r['year']})" for _, r in results.iterrows()
                    )
                    answer_full = answer + f"\n\n**Sources:** {sources}"
                else:
                    answer_full = answer

            st.markdown(answer_full)
        st.session_state.messages.append({"role": "assistant", "content": answer_full})

# ---------------- Search Page ----------------
elif page == "🔍 Paper Search":
    st.title("🔍 Paper Search & Summarization")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search papers by topic, method, or keyword", "")
    with col2:
        category = st.selectbox("Category", ["All"] + sorted(df["category"].unique().tolist()))

    top_k = st.slider("Number of results", 1, 15, 5)

    if query:
        results = engine.search(query, top_k=top_k, category=category)
    else:
        results = df if category == "All" else df[df["category"] == category]
        results = results.head(top_k).assign(score=1.0)

    st.write(f"**{len(results)} paper(s) found**")

    for _, row in results.iterrows():
        with st.expander(f"📄 {row['title']}  ({row['year']}) — {row['category']}"):
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**arXiv ID:** {row['arxiv_id']}")
            st.markdown(f"**Keywords:** {row['keywords']}")
            st.markdown("**Summary:**")
            st.write(row["summary"])
            sentences_n = st.slider(
                "Extractive summary length (sentences)", 1, 3, 1,
                key=f"sumlen_{row['id']}"
            )
            st.markdown("**Auto-generated extractive summary:**")
            st.info(extractive_summarize(row["summary"], n_sentences=sentences_n))

    if len(results) > 1:
        st.markdown("---")
        st.subheader("📋 Combined summary of results")
        st.markdown(multi_doc_summary(results))

# ---------------- Visualization Page ----------------
elif page == "📊 Concept Visualization":
    st.title("📊 Concept Visualization")

    tab1, tab2 = st.tabs(["Category distribution", "Keyword co-occurrence network"])

    with tab1:
        counts = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color="#4C72B0")
        ax.set_ylabel("Number of papers")
        ax.set_title("Papers per category in the knowledge base")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

        year_counts = df["year"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.plot(year_counts.index, year_counts.values, marker="o", color="#DD8452")
        ax2.set_title("Papers per year")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    with tab2:
        st.caption("Shows how keywords co-occur across papers — an approximation of "
                   "concept relationships in the field.")
        focus_cat = st.selectbox("Filter by category", ["All"] + sorted(df["category"].unique().tolist()),
                                  key="graph_cat")
        sub = df if focus_cat == "All" else df[df["category"] == focus_cat]

        G = nx.Graph()
        for kw_str in sub["keywords"]:
            kws = [k.strip() for k in kw_str.split(",")]
            for i in range(len(kws)):
                for j in range(i + 1, len(kws)):
                    if G.has_edge(kws[i], kws[j]):
                        G[kws[i]][kws[j]]["weight"] += 1
                    else:
                        G.add_edge(kws[i], kws[j], weight=1)

        if len(G.nodes) == 0:
            st.warning("No keywords to display for this filter.")
        else:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            pos = nx.spring_layout(G, seed=42, k=0.6)
            weights = [G[u][v]["weight"] for u, v in G.edges]
            degrees = dict(G.degree)
            node_sizes = [200 + 120 * degrees[n] for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#55A868", alpha=0.85, ax=ax3)
            nx.draw_networkx_edges(G, pos, width=[0.5 + 0.5 * w for w in weights], alpha=0.4, ax=ax3)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax3)
            ax3.axis("off")
            ax3.set_title(f"Keyword co-occurrence network ({focus_cat})")
            st.pyplot(fig3)

# ---------------- About Page ----------------
else:
    st.title("ℹ️ About this project")
    st.markdown("""
This chatbot is a **domain-expert assistant for computer science / AI research**,
built as a retrieval-augmented system over a curated subset of arXiv papers.

### Architecture
1. **Corpus** — `data/papers.csv`: title, authors, year, category, keywords,
   and an original summary for 50 landmark CS/AI papers (transformers, CNNs,
   RL, diffusion models, embeddings, etc.).
2. **Information retrieval** — TF-IDF vectorization (unigrams + bigrams) with
   cosine similarity (`nlp_engine.ArxivSearchEngine`).
3. **Summarization** — extractive, TextRank-style: builds a sentence
   similarity graph and ranks sentences with PageRank
   (`nlp_engine.extractive_summarize`).
4. **Explanation generation** — retrieval-augmented generation (RAG). Top
   matching papers are retrieved, then either:
   - passed as context to a **local open-source LLM** (Llama 3 / Mistral /
     Phi-3 via [Ollama](https://ollama.com)) if one is running, or
   - composed into an answer with a deterministic template, so the app
     always works with **zero external dependencies**.
5. **UI** — Streamlit, with a chat interface, a paper search + summarization
   panel, and concept-visualization charts (category/year distribution,
   keyword co-occurrence graph).

### Scaling to the full Kaggle arXiv dataset
This build environment has no network access to Kaggle, arXiv's API, or
HuggingFace, so it ships with a 50-paper curated CS/AI knowledge base instead
of the full 1.7M+ paper Kaggle dump. To scale up on your own machine:

```python
# 1. Download from https://www.kaggle.com/datasets/Cornell-University/arxiv
# 2. Filter to your subset, e.g. categories starting with "cs."
import json, pandas as pd
rows = []
with open("arxiv-metadata-oai-snapshot.json") as f:
    for line in f:
        paper = json.loads(line)
        if paper["categories"].startswith("cs."):
            rows.append(paper)
df = pd.DataFrame(rows)
df.to_csv("data/papers_full.csv", index=False)
```
Then point `DATA_PATH` in `app.py` at the new file. The TF-IDF search will
scale to hundreds of thousands of documents; for very large corpora, swap in
a sentence-embedding index (e.g. `sentence-transformers` + FAISS) for faster
approximate nearest-neighbor search — the `ArxivSearchEngine` interface is
designed to be swapped out without touching the rest of the app.

### Enabling a real open-source LLM
Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, then just
launch the app — `llm_backend.py` auto-detects it and routes explanation
generation through it instead of the template fallback.
""")
