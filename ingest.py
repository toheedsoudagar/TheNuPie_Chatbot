# ingest.py
from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_unstructured import UnstructuredLoader 
from langchain_community.vectorstores.utils import filter_complex_metadata 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tempfile import TemporaryDirectory

# OCR Handling
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --------- Configuration ----------
DOCS_DIR = "docs"
DB_DIR = "chroma_db"

# EMBEDDINGS = LOCAL (Stable, Free)
EMBEDDING_MODEL = "embeddinggemma:latest" 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# ----------------------------------

def ocr_pdf_to_documents(pdf_path):
    docs = []
    if not OCR_AVAILABLE:
        print(f"[ingest][ocr] OCR dependencies missing. Skipping OCR for {pdf_path}.")
        return docs

    print(f"[ingest][ocr] Running OCR fallback on {pdf_path} ...")
    try:
        with TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, dpi=200, output_folder=tmpdir)
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": Path(pdf_path).name}))
                except Exception: pass
    except Exception as e:
        print(f"[ingest][ocr] Failed to OCR {pdf_path}: {e}")
    return docs

def load_all_documents():
    docs = []
    path = Path(DOCS_DIR)

    if not path.exists():
        raise ValueError(f"Docs directory {DOCS_DIR} does not exist.")

    for file in sorted(path.iterdir()):
        try:
            suffix = file.suffix.lower()
            if suffix in ['.sql', '.csv', '.xls', '.xlsx']:
                print(f"[ingest] Skipping structured data file: {file.name}")
                continue

            if suffix == ".pdf":
                print(f"[ingest] Loading PDF → {file.name}")
                try:
                    pdf_docs = PyPDFLoader(str(file)).load()
                    non_empty = sum(1 for d in pdf_docs if d.page_content.strip())
                    if non_empty < max(1, len(pdf_docs) // 2):
                        print("[ingest] PDF text sparse — attempting OCR.")
                        ocr_docs = ocr_pdf_to_documents(str(file))
                        if ocr_docs: pdf_docs = ocr_docs
                    for d in pdf_docs: d.metadata["source"] = file.name
                    docs.extend(pdf_docs)
                except Exception:
                    ocr_docs = ocr_pdf_to_documents(str(file))
                    docs.extend(ocr_docs)

            elif suffix == ".txt":
                print(f"[ingest] Loading TXT → {file.name}")
                txt_docs = TextLoader(str(file), encoding='utf-8', autodetect_encoding=True).load()
                for d in txt_docs: d.metadata["source"] = file.name
                docs.extend(txt_docs)

            elif suffix in [".doc", ".docx", ".ppt", ".pptx", ".html", ".htm"]:
                print(f"[ingest] Loading Unstructured → {file.name}")
                other_docs = UnstructuredLoader(str(file)).load()
                for d in other_docs: d.metadata["source"] = file.name
                docs.extend(other_docs)

        except Exception as e:
            print(f"[ingest] Failed to load {file.name}: {e}")

    print(f"[ingest] Loaded {len(docs)} raw document sections.")
    return docs

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")
    return chunks

def build_vectorstore(chunks):
    if not chunks:
        print("[ingest] No chunks to ingest.")
        return None
    
    print(f"[ingest] Building embeddings locally ({EMBEDDING_MODEL})...")
    
    # FIX: No 'headers' or 'base_url' passed here. Defaults to http://localhost:11434
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print("[ingest] Chroma DB built and persisted.")
    return vectordb

def run_ingestion_if_needed():
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("[ingest] No Vector DB found → Running ingestion...")
        docs = load_all_documents()
        if docs:
            chunks = split_into_chunks(docs)
            chunks = filter_complex_metadata(chunks)
            build_vectorstore(chunks)
    else:
        print("[ingest] Vector DB exists → Skipping ingestion.")

if __name__ == "__main__":
    run_ingestion_if_needed()