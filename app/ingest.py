# app/ingest.py
# PURPOSE: Read PDFs → split into chunks → embed → save to ChromaDB
# On Azure: downloads PDFs from Blob Storage first
# Locally: reads from data/ folder directly

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = "vectorstore"
DATA_PATH   = "data"

# ── Blob download — runs on Azure, skipped locally ────────────────────
def download_pdfs_from_blob():
    """
    On Azure: download PDFs from blob storage to local data/ folder.
    Locally: PDFs already exist in data/ — this is skipped.
    """
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not storage_conn:
        print("[Blob] No storage connection string — using local data/ folder")
        return

    os.makedirs(DATA_PATH, exist_ok=True)
    try:
        from azure.storage.blob import BlobServiceClient
        client    = BlobServiceClient.from_connection_string(storage_conn)
        container = client.get_container_client("documents")
        blobs     = list(container.list_blobs())
        print(f"[Blob] Found {len(blobs)} PDFs in blob storage")
        for blob in blobs:
            dest = os.path.join(DATA_PATH, blob.name)
            if not os.path.exists(dest):
                print(f"[Blob] Downloading: {blob.name}")
                with open(dest, "wb") as f:
                    f.write(container.download_blob(blob.name).readall())
                print(f"[Blob] Done: {blob.name}")
            else:
                print(f"[Blob] Already exists: {blob.name}")
    except Exception as e:
        print(f"[Blob] Download failed: {e}")

# ── Load PDFs ─────────────────────────────────────────────────────────
def load_pdfs():
    docs = []
    if not os.path.exists(DATA_PATH):
        print(f"[Ingest] data/ folder not found")
        return docs
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            path   = os.path.join(DATA_PATH, filename)
            loader = PyMuPDFLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"Loaded: {filename} ({len(loaded)} pages)")
    print(f"\nTotal pages loaded: {len(docs)}")
    return docs

# ── Chunk documents ───────────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ── Build vectorstore ─────────────────────────────────────────────────
def build_vectorstore(chunks):
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("Building ChromaDB vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_PATH
    )
    print(f"Vectorstore saved to: {CHROMA_PATH}")
    return vectorstore

if __name__ == "__main__":
    download_pdfs_from_blob()   # no-op locally, downloads on Azure
    docs   = load_pdfs()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("\nIngest complete. Ready to query.")