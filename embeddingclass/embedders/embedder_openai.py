from __future__ import annotations
from typing import List, Sequence, Union, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

from embeddingclass.utils.utils_normalize import l2_normalize
from embeddingclass.embedders.embedder_base import DocLike

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dimension: Optional[int] = None, normalize: bool = True):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.model = model
        self.dimension = dimension
        self.normalize = normalize
        self._emb = OpenAIEmbeddings(model=model)

    def embed_query(self, text: str) -> List[float]:
        v = self._emb.embed_query(text)
        if self.dimension is not None and len(v) != self.dimension:
            raise ValueError(f"Embedding dim mismatch: expected {self.dimension}, got {len(v)}")
        return l2_normalize(v) if self.normalize else v

    def embed_documents(self, documents: Sequence[DocLike]) -> List[List[float]]:
        texts: List[str] = [
            d.page_content if isinstance(d, Document) else str(d)
            for d in documents
        ]
        vecs = self._emb.embed_documents(texts)
        if self.dimension is not None and vecs and len(vecs[0]) != self.dimension:
            raise ValueError(f"Embedding dim mismatch: expected {self.dimension}, got {len(vecs[0])}")
        return [l2_normalize(v) for v in vecs] if self.normalize else vecs
