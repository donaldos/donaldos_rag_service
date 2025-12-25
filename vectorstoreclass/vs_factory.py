# vectorstoreclass/vs_factory.py
from __future__ import annotations
from langchain_core.embeddings import Embeddings
from .vs_base import VectorStoreConfig, VectorStoreDriver

def build_vectorstore(cfg: VectorStoreConfig, embeddings: Embeddings) -> VectorStoreDriver:
    backend = cfg.backend.lower()

    if backend == "chroma":
        from .vs_chroma import ChromaDriver
        return ChromaDriver(cfg, embeddings)

    if backend == "faiss":
        from .vs_faiss import FaissDriver
        return FaissDriver(cfg, embeddings)

    if backend == "pinecone":
        from .vs_pinecone import PineconeDriver
        return PineconeDriver(cfg, embeddings)

    raise ValueError(f"Unsupported vectorstore backend: {cfg.backend}")
