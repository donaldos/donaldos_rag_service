# vectorstoreclass/vs_pinecone.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from .vs_base import VectorStoreConfig, VectorStoreDriver


@dataclass(frozen=True)
class PineconeConfig:
    """
    Pinecone 전용 설정 (VectorStoreConfig에 넣어도 되지만,
    분리해두면 backend별 설정 관리가 깔끔해짐)
    """
    api_key: str
    index_name: str
    namespace: str = ""          # Pinecone namespace (기본은 '')
    text_key: str = "text"       # Document text를 metadata에 저장할 key


class PineconeDriver(VectorStoreDriver):
    """
    Pinecone 기반 VectorStore Driver

    - langchain_pinecone.PineconeVectorStore 사용
    - add_documents: ids 지정 가능(권장: doc_id/chunk_id로 안정적 ID)
    - delete: ids 또는 where(metadata filter) 지원
    """

    def __init__(
        self,
        cfg: VectorStoreConfig,
        embeddings: Embeddings,
        pcfg: PineconeConfig,
        *,
        pool_threads: int = 4,
    ):
        self._cfg = cfg
        self._emb = embeddings
        self._pcfg = pcfg

        # Pinecone SDK init
        # (grpc 사용 시 pinecone[grpc] 설치 후 pinecone.grpc.PineconeGRPC도 가능)
        from pinecone import Pinecone

        pc = Pinecone(api_key=pcfg.api_key)
        index = pc.Index(pcfg.index_name, pool_threads=pool_threads)

        # LangChain Pinecone VectorStore
        from langchain_pinecone import PineconeVectorStore

        self._vs = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key=pcfg.text_key,
            namespace=pcfg.namespace,
        )

    # ---- VectorStoreDriver API ----

    def add_documents(self, docs: Sequence[Document]) -> List[str]:
        # ids를 외부에서 생성해서 넣고 싶다면:
        # return self._vs.add_documents(list(docs), ids=ids)
        return self._vs.add_documents(list(docs))

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        - ids: 벡터 ID 목록 삭제
        - where: metadata 조건 삭제 (Pinecone의 filter로 전달)
        """
        if ids and where:
            raise ValueError("Provide only one of ids or where (filter), not both.")

        if ids:
            # langchain_pinecone: delete(ids=..., namespace=...) 지원 :contentReference[oaicite:1]{index=1}
            self._vs.delete(ids=ids, namespace=self._pcfg.namespace)
            return

        if where:
            # where -> Pinecone filter :contentReference[oaicite:2]{index=2}
            self._vs.delete(filter=where, namespace=self._pcfg.namespace)
            return

        # 아무 조건 없이 호출되면 실수 방지
        raise ValueError("delete requires ids or where(filter).")

    def as_retriever(self, cfg: VectorStoreConfig) -> BaseRetriever:
        search_kwargs: Dict[str, Any] = {"k": cfg.k}
        if cfg.score_threshold is not None:
            search_kwargs["score_threshold"] = cfg.score_threshold

        # namespace는 VectorStore 생성 시 고정해두는 게 운영에서 실수 적음
        return self._vs.as_retriever(
            search_type=cfg.search_type,  # "similarity" | "mmr"
            search_kwargs=search_kwargs,
        )
