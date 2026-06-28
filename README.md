# Dynamic Knowledge Base Expansion — Customer Service Bot

## Overview

This system implements a **real-time Gen AI Customer Service Bot** with a
self-updating knowledge base. It periodically fetches content from specified
web/file sources, embeds new chunks into a **ChromaDB** vector store, and
uses **LangChain + OpenAI** to answer customer questions from the latest data.

---

## Architecture

```
Sources (Web / Files)
        │
        ▼
  Fetch & Scrape
        │
        ▼
  Chunk & Hash  ──► Skip if already seen
        │
        ▼
  Embed & Upsert into ChromaDB
        │
        ▼
  RetrievalQA Chain  ◄──  Customer Question
        │
        ▼
     Answer + Sources
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your key

# 3. Run the bot
python knowledge_base_updater.py
```

---

## How It Works

| Component | Description |
|---|---|
| `fetch_web_content()` | Scrapes a URL, strips boilerplate, returns a Document |
| `fetch_file_content()` | Loads a local `.txt` file as a Document |
| `hash_content()` | MD5 hash per chunk — skips duplicates on re-runs |
| `KnowledgeBaseManager` | Manages ChromaDB, splits docs, upserts new chunks |
| `CustomerServiceBot` | RetrievalQA chain over the live vector store |
| `schedule_updates()` | Background thread polling sources every 60 minutes |

---

## Adding New Sources

Edit the `KNOWLEDGE_SOURCES` list in `knowledge_base_updater.py`:

```python
KNOWLEDGE_SOURCES = [
    {"name": "My FAQ Page",  "url": "https://example.com/faq",  "type": "web"},
    {"name": "Company Docs", "url": "./data/company_info.txt",   "type": "file"},
]
```

---

## Expected Outcome

- The chatbot automatically incorporates new information as sources are updated.
- No manual re-indexing needed — the scheduler handles it.
- Duplicate content is skipped via content hashing.
- Every answer returns which sources it used.
