import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core import load_data, load_lexicon, MedQARetriever, MedicalEntityRecognizer, compose_answer

st.set_page_config(page_title="MedQuAD Medical Q&A Chatbot", page_icon="🩺", layout="wide")


@st.cache_resource
def load_engine():
    df = load_data()
    retriever = MedQARetriever(df)
    lexicon = load_lexicon()
    ner = MedicalEntityRecognizer(lexicon)
    return df, retriever, ner


df, retriever, ner = load_engine()

# ---------------- Sidebar ----------------
st.sidebar.title("🩺 Medical Q&A Chatbot")
st.sidebar.caption("Built on the real MedQuAD dataset")
page = st.sidebar.radio("Navigate", ["💬 Chat", "🔍 Browse Q&A", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.metric("Q&A pairs indexed", len(df))
st.sidebar.markdown(f"**Sources:** {', '.join(sorted(df['source'].unique()))}")

st.sidebar.warning(
    "⚠️ Educational demo only — not medical advice. Always consult a "
    "qualified healthcare provider for medical concerns."
)

# ---------------- Chat Page ----------------
if page == "💬 Chat":
    st.title("🩺 MedQuAD Medical Q&A Chatbot")
    st.caption(
        "Ask a medical question — retrieval finds the closest matching Q&A pairs from the real "
        "MedQuAD dataset (NIH/MedlinePlus/CancerGov/GARD/NINDS/NIDDK/NHLBI/NIH SeniorHealth), and "
        "detected medical terms (symptoms/diseases/treatments) are shown for transparency."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me a medical question — e.g. *\"What are the "
                                              "symptoms of asthma?\"*, *\"How is type 2 diabetes "
                                              "treated?\"*, or *\"What causes migraines?\"*"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching MedQuAD and extracting medical entities..."):
                entities = ner.extract(prompt)
                results = retriever.search(prompt, top_k=3)
                answer = compose_answer(prompt, results, entities)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- Browse Page ----------------
elif page == "🔍 Browse Q&A":
    st.title("🔍 Browse the MedQuAD Knowledge Base")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search by topic or keyword", "")
    with col2:
        qtype = st.selectbox("Question type", ["All"] + sorted(df["qtype"].dropna().unique().tolist()))

    top_k = st.slider("Number of results", 1, 20, 8)

    if query:
        results = retriever.search(query, top_k=top_k, qtype=qtype)
    else:
        results = df if qtype == "All" else df[df["qtype"] == qtype]
        results = results.head(top_k)

    st.write(f"**{len(results)} result(s)**")
    for _, row in results.iterrows():
        with st.expander(f"🩺 {row['question']}"):
            st.markdown(f"**Focus:** {row['focus']}  |  **Source:** {row['source']}  |  **Type:** {row['qtype']}")
            st.write(row["answer"])
            entities = ner.extract(row["question"] + " " + str(row["answer"])[:500])
            notes = [f"{c}: {', '.join(sorted(set(v)))}" for c, v in entities.items() if v]
            if notes:
                st.caption("🏷️ " + " | ".join(notes))

# ---------------- About Page ----------------
else:
    st.title("ℹ️ About this project")
    st.markdown("""
This chatbot is a **specialized medical question-answering system** built on the
real [MedQuAD dataset](https://github.com/abachaa/MedQuAD) (Medical Question
Answering Dataset), created by the NIH's National Library of Medicine team.

### Architecture
1. **Dataset** (`app/build_dataset.py`) — clones/parses the real MedQuAD XML
   files (question/answer pairs from NIH, MedlinePlus, CancerGov, GARD,
   NINDS, NIDDK, NHLBI, NIH SeniorHealth) into `data/medquad.csv`. This demo
   indexes 3,000 Q&A pairs (capped per source for a manageable demo size —
   the full dataset has 47,000+ pairs across 12 sources).
2. **Retrieval mechanism** (`app/core.py: MedQARetriever`) — TF-IDF
   vectorization (unigrams + bigrams) with cosine similarity search over
   question/focus text, so a user's question is matched against the closest
   real MedQuAD questions.
3. **Medical entity recognition** (`app/core.py: MedicalEntityRecognizer`)
   — lexicon-based matching against curated lists of common symptoms,
   diseases, and treatments (`app/medical_lexicon.json`), shown alongside
   every answer for transparency.
4. **UI** — Streamlit, with a chat interface and a browse/filter panel
   (filterable by question type: symptoms, treatment, causes, etc.)

### Why lexicon-based NER instead of a trained clinical NER model
This build environment has no internet access to download NER models (e.g.
scispaCy's `en_ner_bc5cdr_md`, or MedCAT). A curated keyword lexicon is a
transparent, dependency-free way to demonstrate the entity-recognition
requirement; swapping in a trained clinical NER model is a drop-in
replacement for `MedicalEntityRecognizer.extract()` — the rest of the app
doesn't need to change.

### Important disclaimer
This is an educational/demo project, **not a medical device and not medical
advice**. Answers come directly from the MedQuAD dataset's own
NIH/MedlinePlus-sourced content, but any real deployment would need
clinical review, a much larger validated dataset, and clear disclaimers
directing users to consult a healthcare professional.
""")
