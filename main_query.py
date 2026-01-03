import time
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 여러분 프로젝트에 이미 있는 것들
# - create_embedder, LCEmbeddingAdapter
# - VectorStoreConfig, build_vectorstore
# - vs.as_retriever(cfg) 가 동작하는 구조라고 가정
# from your_project.embedding import create_embedder, LCEmbeddingAdapter
# from your_project.vectorstore import VectorStoreConfig, build_vectorstore
from embeddingclass import LCEmbeddingAdapter
from embeddingclass import create_embedder
from embeddingclass import EmbedderBase
from vectorstoreclass import VectorStoreConfig, build_vectorstore
from chunkingclass import CSemanticTextSplitter

def build_embedding():
    """FAISS 인덱스 생성 때 사용한 임베딩 설정과 동일해야 함."""
    start = time.time()
    embedder = create_embedder(
        "hf",
        model_name="BAAI/bge-m3",
        device="mps",
        normalize_embeddings=True,
        use_e5_prefix=False,
    )
    lc_emb = LCEmbeddingAdapter(embedder)
    print(f"✅ 임베더 로드/초기화 {time.time()-start:.2f}초")
    return lc_emb


def load_vectorstore(lc_emb):
    """로컬 persist_dir에 저장된 FAISS 인덱스를 로드."""
    start = time.time()
    cfg = VectorStoreConfig(
        backend="faiss",
        persist_dir="./vs_faiss",   # ✅ 기존에 만들어진 로컬 인덱스 경로
        collection="my_docs",
        k=5,
        search_type="similarity",
    )
    vs = build_vectorstore(cfg, lc_emb)  # build_vectorstore가 로드를 지원해야 함
    print(f"✅ VectorStore 로드 {time.time()-start:.2f}초")
    return vs, cfg


def build_rag_chain(retriever):
    """RAG 체인 구성 (프롬프트 + LLM + 출력 파서)."""
    start = time.time()

    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Answer in Korean.

#Context:
{context}

#Question:
{question}

#Answer:
"""
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"✅ RAG Chain 구성 {time.time()-start:.2f}초")
    return chain


def query_once(chain, question: str):
    """질의 실행(= retrieve + prompt + LLM)."""
    start = time.time()
    answer = chain.invoke(question)
    print(f"✅ 질의 실행 {time.time()-start:.2f}초")
    return answer


if __name__ == "__main__":
    load_dotenv()

    query_text = "자사주 매입이란?"

    # 1) 임베더 준비 (검색에도 필요)
    lc_emb = build_embedding()

    # 2) 로컬 FAISS VectorStore 로드
    vs, cfg = load_vectorstore(lc_emb)

    # 3) Retriever 준비
    start = time.time()
    retriever = vs.as_retriever(cfg)
    print(f"✅ Retriever 준비 {time.time()-start:.2f}초")

    # 4) RAG 체인 구성
    chain = build_rag_chain(retriever)

    # 5) 질의 실행
    answer = query_once(chain, query_text)
    print("\n=== Answer ===")
    print(answer)
