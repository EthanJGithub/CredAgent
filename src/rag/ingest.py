"""RAG ingestion — build the CFPB regulatory vector store.

Run once:
    python -m src.rag.ingest

Ensures the corpus exists (downloading real CFPB PDFs where possible, otherwise
writing authoritative regulatory text), loads every ``.pdf`` and ``.txt`` in
``docs/cfpb/``, chunks them, and indexes them into a persistent ChromaDB store
using the local ONNX ``all-MiniLM-L6-v2`` embedding model (no API cost).
"""
import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.corpus import ensure_corpus
from src.rag.embeddings import (
    COLLECTION_NAME, VECTORSTORE_PATH, get_embedding_function,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PDF_DIR = "docs/cfpb"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _load_documents():
    """Return list of (text, source, page) tuples from all corpus files."""
    docs = []
    pdf_paths = sorted(Path(PDF_DIR).glob("*.pdf"))
    txt_paths = sorted(Path(PDF_DIR).glob("*.txt"))

    for pdf_path in pdf_paths:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            pages = PyPDFLoader(str(pdf_path)).load()
            for p in pages:
                docs.append((p.page_content, pdf_path.name, p.metadata.get("page", 0)))
            logger.info("Loaded %s (%d pages).", pdf_path.name, len(pages))
        except Exception as exc:
            logger.warning("Could not load %s: %s", pdf_path.name, exc)

    for txt_path in txt_paths:
        text = txt_path.read_text(encoding="utf-8")
        docs.append((text, txt_path.name, 0))
        logger.info("Loaded %s (%d chars).", txt_path.name, len(text))

    return docs


def ingest():
    ensure_corpus()
    raw_docs = _load_documents()
    if not raw_docs:
        raise FileNotFoundError(f"No corpus files found in {PDF_DIR}/.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
    )

    chunks, metadatas, ids = [], [], []
    for text, source, page in raw_docs:
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(piece)
            metadatas.append({"source": source, "page": page})
            ids.append(f"{Path(source).stem}_{page}_{i}")
    logger.info("Produced %d chunks from %d documents.", len(chunks), len(raw_docs))

    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    batch = 100
    for i in range(0, len(chunks), batch):
        collection.add(
            documents=chunks[i:i + batch],
            metadatas=metadatas[i:i + batch],
            ids=ids[i:i + batch],
        )
        logger.info("Indexed %d/%d chunks.", min(i + batch, len(chunks)), len(chunks))

    logger.info("Vector store built at %s/ — '%s' has %d documents.",
                VECTORSTORE_PATH, COLLECTION_NAME, collection.count())


if __name__ == "__main__":
    ingest()
