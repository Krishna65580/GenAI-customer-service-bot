"""
app.py
------
Streamlit UI for the Medical Q&A Chatbot built on the MedQuAD dataset.

Run with:
    streamlit run app.py
"""
import streamlit as st

from core import EntityRecognizer, Retriever, load_data

DATA_PATH = "data/medquad.csv"

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------
# Cached resources: loaded once per session, not on every rerun.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MedQuAD dataset and building index...")
def get_resources():
    df = load_data(DATA_PATH)
    retriever = Retriever(df)
    recognizer = EntityRecognizer(df["focus"].dropna().unique().tolist())
    return df, retriever, recognizer


df, retriever, recognizer = get_resources()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🩺 About")
    st.write(
        "This chatbot answers medical questions using the "
        "[MedQuAD dataset](https://github.com/abachaa/MedQuAD) — "
        f"**{len(df):,}** question/answer pairs sourced from NIH, CDC, "
        "and other trusted medical sites."
    )
    st.divider()
    top_k = st.slider("Number of answers to show", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption(
        "⚠️ This tool is for educational purposes only and is **not** a "
        "substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )
    st.divider()
    st.subheader("Try an example")
    examples = [
        "What are the symptoms of diabetes?",
        "How is asthma treated?",
        "What causes high blood pressure?",
        "What is Paget's disease of bone?",
        "How to diagnose migraine?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("🩺 Medical Q&A Chatbot")
st.caption("Ask a medical question in plain English — powered by MedQuAD retrieval + entity recognition.")

query = st.text_input(
    "Ask your medical question:",
    key="query_input",
    placeholder="e.g. What are the treatments for Paget's disease of bone?",
)

if query:
    entities = recognizer.extract(query)

    # Detected entities row
    with st.container(border=True):
        st.markdown("**🔍 Detected medical entities**")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Diseases**")
            st.write(", ".join(entities.diseases) if entities.diseases else "_none detected_")
        with cols[1]:
            st.markdown("**Symptoms**")
            st.write(", ".join(entities.symptoms) if entities.symptoms else "_none detected_")
        with cols[2]:
            st.markdown("**Treatments**")
            st.write(", ".join(entities.treatments) if entities.treatments else "_none detected_")

    st.divider()

    results = retriever.search(query, top_k=top_k)

    if results.empty:
        st.warning("I couldn't find a confident match for that question. Try rephrasing it.")
    else:
        st.markdown(f"**💬 Top {len(results)} matching answer(s):**")
        for _, row in results.iterrows():
            confidence = row["similarity"] * 100
            with st.expander(
                f"**{row['question'].strip()}**  —  {confidence:.0f}% match",
                expanded=(row.name == results.index[0]),
            ):
                st.write(row["answer"])
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"📌 Focus: {row['focus']}")
                meta_cols[1].caption(f"🏷️ Category: {row['qtype']}")
                meta_cols[2].caption(f"🔗 Source: {row['source']}")
                if isinstance(row["url"], str) and row["url"].startswith("http"):
                    st.markdown(f"[View original source]({row['url']})")
else:
    st.info("👆 Type a question above, or pick an example from the sidebar to get started.")
