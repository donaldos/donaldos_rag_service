from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 예시: OpenAI 임베딩 (너의 embeddingclass로 교체 가능)
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


from vectorstoreclass import VectorStoreConfig, build_vectorstore
from vectorstoreclass.util import attach_doc_and_chunk_ids, build_where_filter


def load_and_chunk_txt(
    filepath: str,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 10,
) -> list[Document]:
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()  # 보통 길면 1개의 Document로 들어옴

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 메타데이터에 source 넣어두면 삭제/필터링에 유리
    for d in chunks:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = filepath

    # doc_id/chunk_id/vector_id 규약 적용 + metadata 정규화
    chunks = attach_doc_and_chunk_ids(
        chunks,
        default_source=filepath,
        version="v1",   # 문서 버전 바뀌면 "v2"로 바꾸면 doc_id가 달라져서 관리 쉬움
    )
    return chunks


def main():
    load_dotenv()
    # ----------------------------
    # 1) 입력
    # ----------------------------
    filepath = "./data/input.txt"
    chunks = load_and_chunk_txt(filepath)

    # ----------------------------
    # 2) Embeddings
    # ----------------------------
    # OPENAI_API_KEY 환경변수 필요
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        # normalize_embeddings 옵션은 버전에 따라 다를 수 있음
    )

    # ----------------------------
    # 3) VectorStore 생성/로드
    # ----------------------------
    # backend 교체: "faiss" | "chroma" | "pinecone"
    cfg = VectorStoreConfig(
        backend="faiss",
        persist_dir="./vs_faiss",   # FAISS/Chroma는 로컬 저장 경로
        collection="my_docs",       # Chroma/Pinecone에서 주로 사용
        k=5,
        search_type="similarity",
    )

    vs = build_vectorstore(cfg, embeddings)

    # ----------------------------
    # 4) 인덱싱(add)
    # ----------------------------
    ids = vs.add_documents(chunks)
    print(f"Indexed chunks: {len(ids)}")

    # ----------------------------
    # 5) Retriever
    # ----------------------------
    retriever = vs.as_retriever(cfg)

    query = "단어를 데이터베이스에 저장할때 어떤 단위로 저장해?"
#    results = retriever.get_relevant_documents(query)
    results = retriever.invoke(query)
    

    print("\n=== Top Results ===")
    for i, d in enumerate(results, start=1):
        score_info = ""  # retriever에 따라 score가 metadata로 안 내려올 수도 있음
        print(f"\n[{i}] source={d.metadata.get('source')} doc_id={d.metadata.get('doc_id')} chunk_id={d.metadata.get('chunk_id')}{score_info}")
        print(d.page_content[:500])

    # ----------------------------
    # 6) (옵션) 문서 단위 삭제 예시
    # ----------------------------
    # 같은 파일 전체를 지우고 싶다면 where/filter를 사용할 수 있는 backend(Chroma/Pinecone)에서:
    # where = build_where_filter(source=filepath)
    # vs.delete(where=where)

    # Pinecone/Chroma가 아니라 FAISS면 where 삭제가 어렵고,
    # 안정적인 운영을 하려면 vector_id 목록을 별도로 저장/관리하는 패턴이 필요함.


if __name__ == "__main__":
    main()
