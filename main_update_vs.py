import time
from dotenv import load_dotenv
from typing import Union, Sequence, List
from loggerproc import setup_logger
import os
logger = setup_logger("rag_service")

"""
logger.debug("디버그 메시지")
logger.info("문서 로드 완료")
logger.warning("청크 크기가 너무 큽니다")
logger.error("벡터 저장 실패")
logger.exception("예외 발생")
"""

"""
Factory 패턴
파일 확장자(pdf, txt, docx 등)에 따라
 - 내부적으로 적절한 Loader 선택
 결과는 항상 List[Document]
→ 상위 로직은 파일 종류를 신경 쓰지 않음
"""
from loaddocuclass import CDocumentLoaderFactory

"""
다양한 Chunk Splitter 구현체들
공통점:
 - 모두 CBaseChunkSplitter 인터페이스를 따름
현재 실제로 쓰는 것은:
 - CSemanticTextSplitter
→ 전략 패턴 (Strategy Pattern)
"""
from chunkingclass import ( 
    CBaseChunkSplitter, 
    CCharTextSplitter, 
    CRecursiveCharTextSplitter, 
    CTiktokenTextSplitter, 
    CTokenTextSplitter, 
    CSpacyTextSplitter, 
    CSentenceTransformersTokenTextSplitter,
    CNLTKTextSplitter,                          # Not completed
    CKONLPTextSplitter,                         # Not completed     
    CGPT2TokenizerFast,                         # Not completed
    CSemanticTextSplitter,
    CClauseTextSplitter,
    CHeaderTextSplitter,
)

"""
create_embedder
 - HuggingFace / OpenAI / 기타 모델 생성 팩토리

EmbedderBase
 - embedder의 공통 인터페이스

LCEmbeddingAdapter
 - LangChain VectorStore가 요구하는 Embeddings 인터페이스로 변환

👉 “임베딩 엔진”과 “VectorStore”를 느슨하게 결합
"""
from embeddingclass import LCEmbeddingAdapter
from embeddingclass import create_embedder
from embeddingclass import EmbedderBase

"""
VectorStoreConfig
 - backend / persist_dir / search 방식 등 설정 객체

build_vectorstore
 - 설정 + embedding을 받아 실제 VectorStore 생성

Document
 - LangChain 표준 문서 타입
"""
from vectorstoreclass import VectorStoreConfig, build_vectorstore
from langchain_core.documents import Document

"""
타입별칭설정 - 입력유연성 확보
- 청킹 단게에서 문자열리스트, Document리스트를 받을 수 있도록 함
→ 재사용성 증가
"""
DocLike = Union[str, Document]
import sys
from chunking_splitter_reg import create_splitter



def build_indexing_context(chunking_type: str, embedder_type: str, backend: str):
    load_dotenv()

    splitter = create_splitter(
        chunking_type,               # splitter type
        #chunk_size=500,                   # chunk size
        #chunk_overlap=100,                # chunk overlap
    )

    # 2) 임베딩 엔진 준비: 임베더 생성
    if embedder_type == "hf":
        embedder = create_embedder(
            "hf",                           # hf / openai / other
            model_name="BAAI/bge-m3",       # 모델명
            device="mps",                   # cpu/gpu/mps
            normalize_embeddings=True,      # cosine similarity 최적화   
            use_e5_prefix=False,
        )
    elif embedder_type == "openai":
        embedder = create_embedder(
            "openai",                 # "hf" → "openai"
            model_name="text-embedding-3-large",            
            api_key=os.environ["OPENAI_API_KEY"],
    )
    # LangChain VectorStore가 요구하는 Embeddings 인터페이스로 변환
    # 👉 Adapter Pattern
    lc_emb = LCEmbeddingAdapter(embedder)

    # 3) VectorStore 준비: VectorStoreConfig 생성
    persist_dir = "./vectorstore/vs_" + backend + "_" + embedder_type + "_" + chunking_type
    cfg = VectorStoreConfig(
        backend=backend,                # faiss / chroma / pinecone
        persist_dir=persist_dir,       # 로컬 저장 경로: 디스크에 인덱스 저장
        collection="my_docs",           # Chroma/Pinecone에서 주로 사용
        k=5,                            # 검색할 문서 수
        search_type="similarity",       # similarity / exact
    )
    # 설정 + 임베딩을 결합해 VectorStore 인스턴스 생성
    vs = build_vectorstore(cfg, lc_emb)

    return vs, splitter

def run_chunking(splitter: CBaseChunkSplitter, docs: Sequence[DocLike]):
    
    if docs and isinstance(docs[0], Document):
        texts = [doc.page_content for doc in docs]
    else:
        texts = docs

    texts = splitter.create_document(texts)

    return texts

def run_embedding(embedder: EmbedderBase, chunks: List[Document]) -> List[List[float]]:
    chunk_texts = [c.page_content for c in chunks]
    vectors = embedder.embed_documents(chunk_texts)
    return vectors

def index_one_file(vs, splitter, doc_path: str):
    # load → chunk → add_documents
    docs = CDocumentLoaderFactory().load(doc_path)
    texts = run_chunking(splitter, docs)
    ids = vs.add_documents(texts)
    return ids


if __name__ == "__main__":

    # 1) 청킹 단계 준비: ChunkSplitter 생성
    #chunking_types = ["char","recursive_char","tiktoken","token","spacy","sentence_transformers","semantic","clause","header"]
    chunking_types = ["char","recursive_char","tiktoken","token","spacy","semantic","clause","header"]
    embedder_types = ["hf","openai"]
    beckend_types = ["faiss","chroma"] #,"pinecone"]

    for chunking_type in chunking_types:
        for embedder_type in embedder_types:
            for beckend_type in beckend_types:
                logger.info(f"청킹 타입: {chunking_type}\t임베딩 타입: {embedder_type}\t벡터 저장 타입: {beckend_type}") 
                vs, splitter = build_indexing_context(chunking_type, embedder_type, beckend_type)
                files = [
                    "./data/SPRI_Report.pdf",
                    "./data/input.txt",
                    "./data/finance.txt",
                    "./data/2019_01_stockconsert_databook.pdf",
                ]

                for f in files:
                    start = time.time()
                    ids = index_one_file(vs, splitter, f)
                    logger.info(f"✅ {f} 인덱싱 완료: {len(ids)}개\t{time.time()-start:.2f}초")    

    

    
    