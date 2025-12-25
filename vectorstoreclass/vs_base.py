# vectorstoreclass/vs_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Optional, Any, Dict, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

@dataclass(frozen=True)
class VectorStoreConfig:
    backend: str                 # "faiss" | "chroma" | "pinecone" ...
    persist_dir: Optional[str] = None
    collection: Optional[str] = None
    k: int = 5
    search_type: str = "similarity"   # "similarity" | "mmr"
    score_threshold: Optional[float] = None

class VectorStoreDriver(Protocol):
    """VectorStore 구현체 교체를 위한 포트(Driver)"""
    def add_documents(self, docs: Sequence[Document]) -> List[str]: ...
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None: ...
    def as_retriever(self, cfg: VectorStoreConfig) -> BaseRetriever: ...
