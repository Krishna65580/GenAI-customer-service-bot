# Project Report: Dynamic Knowledge Base Expansion for Real-Time Gen AI Customer Service Bot

**Submitted by:** [Your Name]
**Date:** June 28, 2026
**Project:** Learn To Build A Real Time Gen AI Customer Service Bot
**Task:** Implement a system for dynamically expanding the chatbot's knowledge base

---

## 1. Objective

The goal of this task was to implement a mechanism that periodically updates a
vector database with new information from specified sources, enabling a customer
service chatbot to automatically incorporate new knowledge into its responses
over time — without requiring manual re-indexing.

---

## 2. System Architecture

```
┌─────────────────────────────────┐
│     Knowledge Sources           │
│  (Web URLs / Local Text Files)  │
└────────────┬────────────────────┘
             │ fetch every 60 min
             ▼
┌─────────────────────────────────┐
│   Content Fetcher & Scraper     │
│  (requests + BeautifulSoup)     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Text Splitter (LangChain)     │
│  chunk_size=800, overlap=100    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Hash Filter (MD5)             │
│  Skip already-indexed chunks    │
└────────────┬────────────────────┘
             │ new chunks only
             ▼
┌─────────────────────────────────┐
│   ChromaDB Vector Store         │
│  (Persistent, OpenAI Embeddings)│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   RetrievalQA Chain             │
│  (LangChain + GPT-3.5-Turbo)   │
└────────────┬────────────────────┘
             │
             ▼
      Customer Answer + Sources
```

---

## 3. Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.10+                        |
| LLM Framework    | LangChain 0.2                       |
| LLM Model        | OpenAI GPT-3.5-Turbo                |
| Embeddings       | OpenAI text-embedding-ada-002       |
| Vector Database  | ChromaDB (persistent local storage) |
| Web Scraping     | requests + BeautifulSoup4           |
| Scheduling       | schedule (Python library)           |
| Environment      | python-dotenv                       |

---

## 4. Key Components

### 4.1 KnowledgeBaseManager
- Initialises and persists a ChromaDB vector store on disk.
- Fetches documents from configured web/file sources.
- Splits documents into 800-token chunks with 100-token overlap.
- Uses MD5 hashing to avoid re-indexing duplicate content across runs.
- Persists updated embeddings after each ingestion cycle.

### 4.2 CustomerServiceBot
- Wraps a LangChain `RetrievalQA` chain with a custom customer-service prompt.
- Retrieves the top-4 most relevant chunks per query.
- Returns the LLM-generated answer along with source attribution.

### 4.3 Dynamic Scheduler
- Runs in a background daemon thread using Python's `schedule` library.
- Polls all configured sources every 60 minutes (configurable).
- Performs an initial update immediately on startup.

### 4.4 Dataset
- A curated `customer_service_dataset.txt` containing FAQs, product information,
  and company policies was used as the seed knowledge base.
- Additional web sources (company blog, news pages) are polled for live updates.

---

## 5. How Dynamic Expansion Works

1. On startup, all sources are fetched and indexed into ChromaDB.
2. Every 60 minutes, the scheduler re-fetches all sources.
3. Each chunk's MD5 hash is checked against a persistent `seen_hashes.txt`.
4. Only genuinely new chunks are embedded and upserted — no duplicates.
5. The chatbot's retriever always queries the latest state of the vector store,
   meaning new information is immediately available to answer customer queries.

---

## 6. Expected Outcome — Achieved

✅ Chatbot automatically incorporates new information from configured sources.
✅ Vector database is updated periodically without manual intervention.
✅ Duplicate content is efficiently filtered using content hashing.
✅ Every answer is grounded in retrieved context with source attribution.
✅ System is extensible — add any URL or text file as a new source.

---

## 7. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Add OPENAI_API_KEY to .env

# Run the bot
python knowledge_base_updater.py
```

---

## 8. Sample Interaction

```
You: What is the return policy?
Bot: We offer a 30-day return policy on all products. Items must be unused
     and in their original packaging. To initiate a return, contact
     support@company.com with your order number.
     [Sources: Company FAQ]

You: Tell me about the SmartHome Hub X1.
Bot: The SmartHome Hub X1 is a central smart home controller compatible with
     over 200 devices. It supports Alexa, Google Home, and Apple HomeKit,
     and is priced at $149.99 with a 2-year warranty.
     [Sources: Product Knowledge Base]
```

---

## 9. Future Improvements

- Add RSS feed support as an additional source type.
- Implement a REST API (FastAPI) for production deployment.
- Add source prioritisation and freshness scoring.
- Support PDF and DOCX ingestion for richer knowledge bases.
- Add automated tests for the ingestion pipeline.
