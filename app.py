import os, time, hashlib, logging, schedule, requests
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
CHROMA_DB_PATH = "./chroma_db"
SEEN_HASHES_FILE = "./seen_hashes.txt"
UPDATE_INTERVAL_MINUTES = 60
KNOWLEDGE_SOURCES = [
    {"name": "Anthropic News", "url": "https://www.anthropic.com/news", "type": "web"},
    {"name": "Company FAQ", "url": "./customer_service_dataset.txt", "type": "file"},
]
PROMPT = PromptTemplate(input_variables=["context", "question"],
    template="You are a helpful customer service assistant.\nContext: {context}\nQuestion: {question}\nAnswer:")
def load_seen_hashes():
    if not os.path.exists(SEEN_HASHES_FILE): return set()
    with open(SEEN_HASHES_FILE) as f: return set(l.strip() for l in f if l.strip())
def save_hash(h):
    with open(SEEN_HASHES_FILE, "a") as f: f.write(h + "\n")
def hash_content(text): return hashlib.md5(text.encode()).hexdigest()
def fetch_web(url, name):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer"]): tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return [Document(page_content=text, metadata={"source": url, "source_name": name})] if text else []
    except Exception as e:
        logger.error(f"Error: {e}"); return []
def fetch_file(path, name):
    try:
        docs = TextLoader(path).load()
        for d in docs: d.metadata["source_name"] = name
        return docs
    except Exception as e:
        logger.error(f"Error: {e}"); return []
class KnowledgeBaseManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
        self.seen = load_seen_hashes()
    def update(self, sources):
        logger.info("Updating knowledge base...")
        total = 0
        for s in sources:
            docs = fetch_web(s["url"], s["name"]) if s["type"]=="web" else fetch_file(s["url"], s["name"])
            chunks = self.splitter.split_documents(docs)
            new = [c for c in chunks if hash_content(c.page_content) not in self.seen]
            for c in new:
                h = hash_content(c.page_content)
                self.seen.add(h); save_hash(h)
            if new: self.db.add_documents(new); logger.info(f"  Added {len(new)} chunks from {s['name']}")
            total += len(new)
        logger.info(f"Done. {total} new chunks added.\n")
    def retriever(self): return self.db.as_retriever(search_kwargs={"k": 4})
class CustomerServiceBot:
    def __init__(self, kb):
        self.chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY")),
            retriever=kb.retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True)
    def answer(self, q):
        result = self.chain({"query": q})
        sources = list({d.metadata.get("source_name","Unknown") for d in result.get("source_documents",[])})
        return result["result"], sources
def run_scheduler(kb):
    kb.update(KNOWLEDGE_SOURCES)
    schedule.every(UPDATE_INTERVAL_MINUTES).minutes.do(kb.update, KNOWLEDGE_SOURCES)
    while True: schedule.run_pending(); time.sleep(30)
if __name__ == "__main__":
    import threading
    kb = KnowledgeBaseManager()
    threading.Thread(target=run_scheduler, args=(kb,), daemon=True).start()
    time.sleep(5)
    bot = CustomerServiceBot(kb)
    print("\n🤖 Customer Service Bot — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit","exit"): break
        if q:
            ans, src = bot.answer(q)
            print(f"\nBot: {ans}\n[Sources: {', '.join(src)}]\n")
