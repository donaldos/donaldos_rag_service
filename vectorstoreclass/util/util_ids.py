# vectorstoreclass/util/util_ids.py
from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional
from langchain_core.documents import Document


def stable_hash(*parts: str, algo: str = "sha1", size: int = 16) -> str:
    """
    문자열 파트들을 합쳐 안정적 해시 생성.
    - size: 반환 해시 문자열 길이(짧게 잘라 사용)
    """
    h = hashlib.new(algo)
    for p in parts:
        if p is None:
            p = ""
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")  # separator (unit separator)
    return h.hexdigest()[:size]


def make_doc_id(
    *,
    source: str,
    source_id: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """
    문서 단위 ID.
    - 권장: source(파일경로/URL/문서키) + source_id(있으면) + version(있으면)
    """
    return stable_hash("doc", source, source_id or "", version or "")


def make_chunk_id(*, doc_id: str, chunk_index: int) -> str:
    """
    청크 단위 ID (문서 내 순서 기반)
    """
    return stable_hash("chunk", doc_id, str(chunk_index))


def make_vector_id(*, doc_id: str, chunk_id: str) -> str:
    """
    벡터(레코드) ID.
    - Pinecone/Chroma 등에 넣을 ids로 사용하면 '문서 교체(삭제→재삽입)'가 쉬워짐.
    """
    return f"{doc_id}:{chunk_id}"


def ensure_ids_for_documents(
    docs: Iterable[Document],
    *,
    default_source: str = "unknown",
    version: Optional[str] = None,
    chunk_index_start: int = 0,
) -> List[Document]:
    """
    Document.metadata에 doc_id/chunk_id/vector_id가 없으면 채워줌.
    - 입력 docs는 순서가 chunk_index로 사용됨(=splitter 결과 순서)
    - 반환은 '수정된 동일 객체' 리스트(얕은 변경)
    """
    out: List[Document] = []
    for i, d in enumerate(docs, start=chunk_index_start):
        md = dict(d.metadata or {})
        source = str(md.get("source") or default_source)
        source_id = md.get("source_id")
        doc_id = md.get("doc_id") or make_doc_id(source=source, source_id=str(source_id) if source_id else None, version=version)
        chunk_id = md.get("chunk_id") or make_chunk_id(doc_id=doc_id, chunk_index=i)
        vector_id = md.get("vector_id") or make_vector_id(doc_id=doc_id, chunk_id=chunk_id)

        md.update({"source": source, "doc_id": doc_id, "chunk_id": chunk_id, "vector_id": vector_id})
        d.metadata = md
        out.append(d)
    return out
