"""Shared embedding function for ingestion and retrieval.

Uses ChromaDB's built-in ONNX ``all-MiniLM-L6-v2`` model. This is the same
model the project spec calls for, but the ONNX runtime build avoids pulling in
PyTorch (~2 GB) — keeping installs light and Streamlit Cloud deploys fast.

Ingestion and retrieval MUST use the same function, so both import from here.
"""
from chromadb.utils import embedding_functions

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "cfpb_regulations"
VECTORSTORE_PATH = "vectorstore"


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()
