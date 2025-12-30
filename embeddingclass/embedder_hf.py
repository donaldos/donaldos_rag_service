from __future__ import annotations
from typing import List, Sequence, Union, Optional
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

from embeddingclass.embedders.embedder_base import DocLike

class HuggingFaceEmbedder:
    """
    - E5 계열이면 query/passage prefix를 권장
    - normalize_embeddings=True면 이미 정규화된 벡터가 나옴
    """
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
