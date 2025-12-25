# vectorstoreclass/util/util_migrate.py
from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document

from .util_metadata import attach_doc_and_chunk_ids


def _doc_to_dict(d: Document) -> dict:
    return {
        "page_content": d.page_content,
        "metadata": dict(d.metadata or {}),
    }


def _dict_to_doc(x: dict) -> Document:
    return Document(
        page_content=x.get("page_content", ""),
        metadata=x.get("metadata", {}) or {},
    )


def export_documents_jsonl(
    docs: Iterable[Document],
    out_path: str,
    *,
    default_source: str = "unknown",
    version: Optional[str] = None,
) -> None:
    """
    Document 목록을 JSONL로 저장(한 줄에 한 Document).
    - 이후 어떤 VectorStore로든 재인덱싱 가능
    """
    prepared = attach_doc_and_chunk_ids(docs, default_source=default_source, version=version)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in prepared:
            f.write(json.dumps(_doc_to_dict(d), ensure_ascii=False) + "\n")


def import_documents_jsonl(
    in_path: str,
    *,
    default_source: str = "unknown",
    version: Optional[str] = None,
) -> List[Document]:
    """
    JSONL에서 Document 목록 로드.
    """
    docs: List[Document] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(_dict_to_doc(json.loads(line)))

    return attach_doc_and_chunk_ids(docs, default_source=default_source, version=version)


def migrate_by_reindexing(
    *,
    docs: Sequence[Document],
    target_vectorstore_driver,
    default_source: str = "unknown",
    version: Optional[str] = None,
) -> List[str]:
    """
    백엔드 교체 시 권장 마이그레이션:
    - (소스 오브 트루스에서) docs를 다시 받아와서
    - 규약 적용 후
    - target_vectorstore_driver.add_documents(...) 로 재인덱싱
    """
    prepared = attach_doc_and_chunk_ids(docs, default_source=default_source, version=version)
    return target_vectorstore_driver.add_documents(prepared)
