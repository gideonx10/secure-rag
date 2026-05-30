# app/ingest.py
# PURPOSE: Read PDFs → split into chunks → embed → save to ChromaDB

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = "vectorstore"
DATA_PATH   = "data"

def load_pdfs():
    docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_PATH, filename)
            loader = PyMuPDFLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"Loaded: {filename} ({len(loaded)} pages)")
    print(f"\nTotal pages loaded: {len(docs)}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

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
    # Chroma auto-persists — no manual save needed
    print(f"Vectorstore saved to: {CHROMA_PATH}")
    return vectorstore

if __name__ == "__main__":
    docs   = load_pdfs()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("\nIngest complete. Ready to query.")