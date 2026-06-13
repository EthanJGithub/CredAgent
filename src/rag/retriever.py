"""Query interface for the CFPB regulatory vector store."""
import logging
from typing import List, Optional

import chromadb

from src.rag.embeddings import (
    COLLECTION_NAME, VECTORSTORE_PATH, get_embedding_function,
)

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
        logger.info("ChromaDB collection loaded (%d docs).", _collection.count())
    return _collection


def retrieve(query: str, n_results: int = 5) -> List[str]:
    """Return the most relevant CFPB policy chunks for a natural-language query."""
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    formatted = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{chunk}")
    return formatted


def is_ready() -> bool:
    """True when the vector store is initialised and queryable."""
    try:
        return _get_collection().count() > 0
    except Exception:
        return False
