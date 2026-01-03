# vectorstoreclass/vs_faiss.py
from __future__ import annotations

import os
from typing import Sequence, Optional, Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from .vs_base import VectorStoreConfig, VectorStoreDriver


class FaissDriver(VectorStoreDriver):
    """
    FAISS 기반 VectorStore Driver (local persistence)
    - persist_dir에 인덱스를 저장/로드
    - 인덱스가 없으면 embeddings로 dimension을 추정해 empty index 생성
    """

    def __init__(
        self,
        cfg: VectorStoreConfig,
        embeddings: Embeddings,
        *,
        index_name: str = "index",
        auto_persist: bool = True,
        allow_dangerous_deserialization: bool = True,
    ):
        if not cfg.persist_dir:
            raise ValueError("FAISS backend requires cfg.persist_dir")

        self._cfg = cfg
        self._emb = embeddings
        self._index_name = index_name
        self._auto_persist = auto_persist
        self._allow_dangerous_deserialization = allow_dangerous_deserialization

        os.makedirs(cfg.persist_dir, exist_ok=True)

        # 1) 있으면 로드
        if self._has_saved_index(cfg.persist_dir, index_name):
            self._vs = FAISS.load_local(
                folder_path=cfg.persist_dir,
                embeddings=embeddings,
                index_name=index_name,
                #allow_dangerous_deserialization=allow_dangerous_deserialization,
                allow_dangerous_deserialization=True,
            )
        else:
            # 2) 없으면 empty index 생성
            self._vs = self._create_empty_faiss(embeddings)

            # 최초 생성도 저장해두면 운영상 편함
            self._save()

    @staticmethod
    def _has_saved_index(persist_dir: str, index_name: str) -> bool:
        # LangChain FAISS는 보통 index_name.faiss + index_name.pkl 형태로 저장됨
        faiss_path = os.path.join(persist_dir, f"{index_name}.faiss")
        pkl_path = os.path.join(persist_dir, f"{index_name}.pkl")
        return os.path.isfile(faiss_path) and os.path.isfile(pkl_path)

    def _create_empty_faiss(self, embeddings: Embeddings) -> FAISS:
        # dimension 추정 (임베딩 모델에 따라 반드시 필요)
        dim = len(embeddings.embed_query("dimension_probe"))

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FAISS backend requires faiss library. Install faiss-cpu or faiss-gpu."
            ) from e

        # 기본은 L2 distance. (normalize_embeddings=True면 IP도 고려 가능)
        index = faiss.IndexFlatL2(dim)

        # LangChain FAISS는 docstore + index_to_docstore_id 매핑을 가짐
        docstore = InMemoryDocstore({})
        index_to_docstore_id: Dict[int, str] = {}

        return FAISS(
            embedding_function=embeddings,
            #embedding=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def _save(self) -> None:
        self._vs.save_local(folder_path=self._cfg.persist_dir, index_name=self._index_name)

    # ---- VectorStoreDriver API ----

    def add_documents(self, docs: Sequence[Document]) -> List[str]:
        #ids = self._vs.add_documents(list(docs))
        ids = self._vs.add_documents(docs)
        if self._auto_persist:
            self._save()
        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        주의:
        - FAISS는 백엔드 특성상 'where(메타데이터 필터)' 기반 삭제를 직접 지원하지 않는 경우가 많음.
        - ids 기반 삭제는 LangChain/FAISS 버전에 따라 지원 범위가 다를 수 있음.
        """
        if where is not None and ids is None:
            raise NotImplementedError(
                "FAISS driver does not support metadata-based delete(where). "
                "Use ids delete or rebuild index."
            )

        if ids:
            # LangChain FAISS는 버전에 따라 delete(ids=...) 지원/미지원이 있을 수 있음
            try:
                self._vs.delete(ids=ids)  # type: ignore
            except Exception as e:
                raise NotImplementedError(
                    "This LangChain FAISS version may not support delete(ids). "
                    "Consider rebuilding the index."
                ) from e

            if self._auto_persist:
                self._save()

    def as_retriever(self, cfg: VectorStoreConfig) -> BaseRetriever:
        search_kwargs = {"k": cfg.k}
        # FAISS retriever는 score_threshold가 버전에 따라 다를 수 있어 옵션으로만 둠
        if cfg.score_threshold is not None:
            search_kwargs["score_threshold"] = cfg.score_threshold

        return self._vs.as_retriever(
            search_type=cfg.search_type,  # "similarity" | "mmr"
            search_kwargs=search_kwargs,
        )
