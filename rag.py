# rag.py
import re
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from ingest import run_ingestion_if_needed
from sql_agent import SQLAgent
from db_setup import create_database_from_sql_files, ANCHOR_DB_URI

# ---------- CONFIGURATION ----------
# 1. Point everything to LOCALHOST (Transparent Proxy)
OLLAMA_URL = "http://127.0.0.1:11434"

# 2. Models
LLM_MODEL = "gpt-oss:120b-cloud"
EMBEDDING_MODEL = "embeddinggemma:latest" 

CHROMA_DB_DIR = "chroma_db"
LLM_TEMPERATURE = 0.0
# -----------------------------------

class RAGPipeline:
    def __init__(self):
        print("[INIT] Setting up isolated SQL databases...")
        self.anchor_db_uri = create_database_from_sql_files(ANCHOR_DB_URI)
        run_ingestion_if_needed()

        # 1. Embeddings (Local)
        print(f"[INIT] Connecting to Embeddings: {EMBEDDING_MODEL}...")
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL
        )

        print("[INIT] Loading persisted Chroma DB...")
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

        # 2. LLM (Proxied)
        print(f"[INIT] Connecting to LLM: {LLM_MODEL}...")
        self.llm = ChatOllama(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_URL
        )

        # 3. SQL Agent
        self.sql_agent = SQLAgent(
            db_uri=self.anchor_db_uri,
            llm_model=LLM_MODEL,
            llm_temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_URL
        )

    def _is_sql_query(self, query: str) -> bool:
        # EXPANDED KEYWORD LIST
        keywords = [
            # Basic Math
            "total", "sum", "average", "count", "max", "min", "calculate", "value",
            # Listing/Tables
            "list", "table", "show me", "how many",
            # Analysis
            "distribution", "breakdown", "percentage", "proportion", "ratio", 
            "trend", "compare", "difference", "highest", "lowest", "rank", 
            "top", "bottom", "common", "popular"
        ]
        return any(kw in query.lower() for kw in keywords)

    def ask(self, query):
        used_sql = False
        
        # --- PATH 1: SQL ---
        if self._is_sql_query(query):
            print(f"[Router] Routing to SQL Agent: {query!r}")
            summary, raw_rows, source_file = self.sql_agent.ask(query)
            
            is_empty = not raw_rows or raw_rows == []
            is_error = "error" in summary.lower() or "no data" in summary.lower()

            if not is_empty and not is_error:
                print("[Router] SQL Agent success.")
                sources = [{"type": "sql", "source": source_file, "content": raw_rows}]
                return summary, sources
            
            print(f"[Router] SQL Agent returned no data/failure. Falling back to RAG...")
            used_sql = True

        # --- PATH 2: RAG ---
        print(f"[Router] Routing to Document RAG: {query!r}")
        try:
            docs = self.db.similarity_search(query, k=4)
        except Exception:
            docs = []

        if not docs:
            fallback = " (SQL also found no data)." if used_sql else "."
            return f"I couldn't find relevant information in the documents{fallback}", []

        context_str = "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs])

        messages = [
            {"role": "system", "content": "You are a helpful RAG assistant. Use ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]

        response = self.llm.invoke(messages).content

        sources = []
        for d in docs:
            preview = re.sub(r'\s+', ' ', d.page_content).strip()[:400] + "..."
            sources.append({
                "type": "text",
                "source": d.metadata.get("source", "unknown"),
                "content": preview
            })

        return response, sources