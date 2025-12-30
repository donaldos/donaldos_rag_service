from __future__ import annotations
from typing import List, Sequence
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from embeddingclass.embedder_base import EmbedderBase, DocLike


class HuggingFaceEmbedder(EmbedderBase):
    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        normalize_embeddings: bool = True,
        use_e5_prefix: bool = False,
    ):
        self.use_e5_prefix = use_e5_prefix
        self._emb = HuggingFaceEmbeddings(
            model=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )

    def _q(self, q: str) -> str:
        return f"query: {q}" if self.use_e5_prefix else q

    def _p(self, p: str) -> str:
        return f"passage: {p}" if self.use_e5_prefix else p

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(self._q(text))

    def embed_documents(self, documents: Sequence[DocLike]) -> List[List[float]]:
        texts: List[str] = [
            d.page_content if isinstance(d, Document) else str(d)
            for d in documents
        ]
        return self._emb.embed_documents([self._p(t) for t in texts])