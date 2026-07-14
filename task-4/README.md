# arXiv CS/AI Domain-Expert Chatbot

A Streamlit chatbot that acts as a domain expert in computer science / AI
research, built on a curated subset of arXiv papers. It can answer questions,
explain concepts, summarize papers, and visualize how concepts relate — all
grounded in retrieval over the paper corpus (a lightweight RAG pipeline).

## Features
- 💬 **Chat** — ask open-ended questions ("How does attention work?",
  "Compare GANs and diffusion models") and get answers grounded in retrieved
  papers, with sources cited.
- 🔍 **Paper search** — TF-IDF semantic search across 50 landmark CS/AI
  papers, with category filters and on-demand extractive summarization.
- 📊 **Concept visualization** — category/year distribution charts and a
  keyword co-occurrence network graph.
- 🧩 **Pluggable open-source LLM** — auto-detects a locally running Ollama
  model (Llama 3, Mistral, Phi-3, ...) for fluent generation; falls back to a
  deterministic extractive/template pipeline if none is available, so the
  app always runs standalone.

## Project structure
```
arxiv_chatbot/
├── app/
│   ├── app.py           # Streamlit UI (chat, search, visualization, about)
│   ├── nlp_engine.py     # TF-IDF search, extractive summarization (TextRank),
│   │                     #   keyword extraction, RAG-style explanation generation
│   └── llm_backend.py    # Pluggable local open-source LLM (Ollama) integration
├── data/
│   └── papers.csv        # Curated CS/AI paper knowledge base (50 papers)
├── requirements.txt
└── README.md
```

## Setup & run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

### Optional: enable a real open-source LLM
1. Install [Ollama](https://ollama.com/download)
2. `ollama pull llama3.2` (or `mistral`, `phi3`, `qwen2.5`, etc.)
3. `ollama serve` (often starts automatically)
4. Launch the app — it auto-detects Ollama at `localhost:11434` and routes
   explanation generation through it instead of the offline template
   fallback. No code changes needed.

## Why a curated 50-paper corpus instead of the full Kaggle dataset?
This project was built in a sandboxed environment with no network access to
Kaggle, arXiv's bulk API, or HuggingFace. Rather than leave the "dataset"
step unimplemented, it ships with a hand-curated knowledge base of 50
landmark, verifiably real CS/AI papers (Transformers, BERT, GPT-3, ResNet,
GANs, diffusion models, RL, embeddings, etc.) with original summaries. The
`ArxivSearchEngine` class in `nlp_engine.py` is dataset-agnostic — pointing
it at a larger CSV (e.g. the full Kaggle `arxiv-metadata-oai-snapshot.json`
filtered to `cs.*` categories) is a one-line change. See the in-app "About"
tab for the exact filtering script.

## NLP techniques implemented
| Task | Technique |
|---|---|
| Information retrieval | TF-IDF (uni+bigrams) + cosine similarity |
| Summarization | Extractive, TextRank (sentence graph + PageRank) |
| Keyword extraction | Per-document TF-IDF top-terms |
| Explanation generation | Retrieval-augmented generation (RAG): retrieved context → open-source LLM (if available) or template composition |

## Known limitations / next steps
- Corpus is 50 papers, not the full 1.7M-paper Kaggle dump — swap in the
  full dataset as described above for production use.
- Without Ollama running, explanations are template-composed from retrieved
  summaries rather than freely generated — still accurate and grounded, but
  less fluent than a full LLM.
- TF-IDF retrieval is lexical; for better semantic matching at scale,
  replace it with sentence-embeddings + a vector index (e.g.
  `sentence-transformers` + FAISS).
