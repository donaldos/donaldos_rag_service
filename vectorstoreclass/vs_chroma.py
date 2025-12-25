# vectorstoreclass/vs_chroma.py
from __future__ import annotations
from typing import Sequence, Optional, Any, Dict, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from .vs_base import VectorStoreConfig, VectorStoreDriver

class ChromaDriver(VectorStoreDriver):
    def __init__(self, cfg: VectorStoreConfig, embeddings: Embeddings):
        self._cfg = cfg
        self._vs = Chroma(
            collection_name=cfg.collection or "default",
            persist_directory=cfg.persist_dir,
            embedding_function=embeddings,
        )

    def add_documents(self, docs: Sequence[Document]) -> List[str]:
        return self._vs.add_documents(list(docs))

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        # Chroma는 where 지원(버전에 따라 차이 있을 수 있음)
        if ids:
            self._vs.delete(ids=ids)
        elif where:
            self._vs.delete(where=where)

    def as_retriever(self, cfg: VectorStoreConfig) -> BaseRetriever:
        search_kwargs = {"k": cfg.k}
        if cfg.score_threshold is not None:
            search_kwargs["score_threshold"] = cfg.score_threshold
        return self._vs.as_retriever(search_type=cfg.search_type, search_kwargs=search_kwargs)
