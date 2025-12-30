from __future__ import annotations
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # 또는 AzureOpenAIEmbeddings 등

from embeddingclass.embedder_base import EmbedderBase, DocLike

class OpenAIEmbedder(EmbedderBase):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url

        self._emb = OpenAIEmbeddings(model=model, **kwargs)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)

    def embed_documents(self, documents: Sequence[DocLike]) -> List[List[float]]:
        texts: List[str] = [
            d.page_content if isinstance(d, Document) else str(d)
            for d in documents
        ]
        return self._emb.embed_documents(texts)
