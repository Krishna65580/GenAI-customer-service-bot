# Medical Q&A Chatbot (MedQuAD)

A specialized medical question-answering chatbot built on the
[MedQuAD dataset](https://github.com/abachaa/MedQuAD) (16,407 QA pairs from
NIH, CDC, GARD, MedlinePlus, and other trusted sources).

## Features

- **Retrieval mechanism** — TF-IDF + cosine similarity over the MedQuAD
  question bank to find the most relevant answer(s) to a user's question.
- **Basic medical entity recognition** — dictionary-based NER that detects
  **diseases** (matched against 5,000+ real condition names from the
  dataset itself), **symptoms**, and **treatments** in the user's query.
- **Simple Streamlit UI** — type a question, see detected entities, and
  browse the top matching answers with their source and confidence score.

## Project structure

```
medqa_bot/
├── app.py              # Streamlit UI
├── core.py             # EntityRecognizer + Retriever classes
├── build_dataset.py     # One-time script: parses raw MedQuAD XML -> data/medquad.csv
├── data/
│   └── medquad.csv      # Flattened dataset (question, answer, focus, source, qtype, url)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the MedQuAD dataset (only needed if you want to rebuild `data/medquad.csv`
   yourself — a pre-built copy is already included in `data/`):

   ```bash
   git clone https://github.com/abachaa/MedQuAD.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Rebuild the dataset CSV from a fresh MedQuAD clone:

   ```bash
   python build_dataset.py --medquad_dir ./MedQuAD --out data/medquad.csv
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

   Then open the URL Streamlit prints (usually `http://localhost:8501`).

## How it works

### Retrieval

`core.Retriever` fits a `TfidfVectorizer` (unigrams + bigrams, English stop
words removed) over each dataset question concatenated with its disease
`focus` term. A user's query is vectorized the same way, and the top-k
most similar questions (by cosine similarity) are returned along with their
answers, source, and category (`qtype`).

### Entity recognition

`core.EntityRecognizer` performs three lightweight, rule-based lookups:

- **Diseases** — substring matching against the ~5,000 unique `focus`
  values already present in MedQuAD (e.g. "Paget's Disease of Bone",
  "Diabetes mellitus type 1"). This gives broad, real-world disease
  coverage without needing an external ontology.
- **Symptoms** / **Treatments** — regex matching against curated keyword
  lists of common symptom and treatment terms (see `SYMPTOM_TERMS` and
  `TREATMENT_TERMS` in `core.py`).

This is intentionally simple (no spaCy/transformers model download
required) so the app starts instantly and runs anywhere with just
`pandas` + `scikit-learn`. If you want higher-precision medical NER later,
swap in `scispaCy`'s `en_ner_bc5cdr_md` model (trained specifically on
diseases and chemicals/drugs) inside `EntityRecognizer.extract`.

## Disclaimer

This tool is for educational purposes only and does not provide medical
advice, diagnosis, or treatment. Always consult a qualified healthcare
provider for medical concerns.
