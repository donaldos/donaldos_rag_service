# vectorstoreclass/util/util_metadata.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from langchain_core.documents import Document

from .util_ids import ensure_ids_for_documents


ALLOWED_META_TYPES = (str, int, float, bool)


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    벡터스토어에 넣기 좋은 형태로 metadata 정규화.
    - Pinecone/Chroma 등은 값 타입 제약이 있는 경우가 많음.
    - dict/list 같은 복잡 타입은 str()로 평탄화(필요시 정책 변경 가능)
    """
    out: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, ALLOWED_META_TYPES):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def ensure_source(docs: Iterable[Document], *, default_source: str = "unknown") -> List[Document]:
    """
    source가 없으면 채움.
    """
    out: List[Document] = []
    for d in docs:
        md = dict(d.metadata or {})
        md["source"] = md.get("source") or default_source
        d.metadata = md
        out.append(d)
    return out


def attach_doc_and_chunk_ids(
    docs: Iterable[Document],
    *,
    default_source: str = "unknown",
    version: Optional[str] = None,
    chunk_index_start: int = 0,
) -> List[Document]:
    """
    source/doc_id/chunk_id/vector_id 규약을 docs에 적용 + metadata 정규화까지.
    """
    docs = ensure_source(docs, default_source=default_source)
    docs = ensure_ids_for_documents(docs, default_source=default_source, version=version, chunk_index_start=chunk_index_start)

    out: List[Document] = []
    for d in docs:
        d.metadata = normalize_metadata(dict(d.metadata or {}))
        out.append(d)
    return out


def build_where_filter(
    *,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    source_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    공통 'where(filter)' 생성기.
    - Chroma의 where, Pinecone의 filter에 그대로 쓸 수 있게 '평평한 dict'로 설계.
    - 복잡 조건($and/$or 등)은 backend별로 다르니 여기서는 단순 동등조건만 권장.
    """
    f: Dict[str, Any] = {}
    if doc_id:
        f["doc_id"] = doc_id
    if source:
        f["source"] = source
    if source_id:
        f["source_id"] = source_id
    if extra:
        for k, v in extra.items():
            if v is not None:
                f[k] = v
    return f
