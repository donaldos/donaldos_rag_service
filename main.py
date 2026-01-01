from loaddocuclass import CDocumentLoaderFactory

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
)

from embeddingclass import LCEmbeddingAdapter
from embeddingclass import create_embedder
from embeddingclass import EmbedderBase

from vectorstoreclass import VectorStoreConfig, build_vectorstore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from typing import Sequence, Union, List
import time

from langchain_core.documents import Document

DocLike = Union[str, Document]

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

if __name__=='__main__':
    load_dotenv()
    DOCFILEPATH = './data/SPRI_Report.pdf'    
    query_text = "구글이 얼마 주기로 했는가?"
    
    '''
    1️⃣ 문서 로드 (Load / Parse)
    docs = CDocumentLoaderFactory().load(DOCFILEPATH)
    파일 → List[Document]
    '''
    start_time = time.time()
    docs = CDocumentLoaderFactory().load(DOCFILEPATH)
    duration = time.time() - start_time
    print(f"1️⃣ 문서 로드: {type(docs)}\t{type(docs[0])}\t{duration:.2f}초")

    # 2️⃣ 텍스트 청킹: texts: List[Document] (청킹 결과)
    '''
    2️⃣ 텍스트 분할/청킹 (Split / Chunk)
    texts = run_chunking(splitter, docs)
    List[Document] → List[Document]
    '''
    start_time = time.time()
    splitter = CSemanticTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = run_chunking(splitter, docs)     # List[Document] 반환
    duration = time.time() - start_time
    print(f"2️⃣ 텍스트 splitt and chunking: {type(texts)}\t{type(texts[0])}\t{duration:.2f}초")  
    
    # 3️⃣ 임베더 생성 (예: HuggingFace)
    '''
    3️⃣ 임베더(모델) 모델 로드/초기화 (Embedder Init / Warmup)
    embedder = create_embedder(...)
    lc_emb = LCEmbeddingAdapter(embedder)
    '''
    start_time = time.time()
    embedder = create_embedder(
        "hf",
        model_name="BAAI/bge-m3",
        device="mps",
        normalize_embeddings=True,
        use_e5_prefix=False,
    )
    lc_emb = LCEmbeddingAdapter(embedder)
    duration = time.time() - start_time
    print(f"3️⃣ 임베더 모델 로드/초기화	{duration:.2f}초")

    # 4️⃣ 문서 임베딩: vectors: List[List[float]] (임베딩 결과)
    '''
    4️⃣ 문서 임베딩 (Chunk Embedding)
    vectors = run_embedding(lc_emb, texts)
    이 이름은 맞는데, 주의할 점:
    지금 코드에서는 vectors를 이후 단계에서 사용하지 않습니다.
    vs.add_documents(texts)가 내부에서 다시 임베딩을 수행한다면 임베딩이 2번 돌 있어요.
    ➡️ 정확히 하려면 둘 중 하나로 통일
    (A) run_embedding 제거하고 add_documents(texts)만 사용
    (B) add_documents 대신 add_embeddings(texts, vectors) 같은 API가 있으면 그걸 사용
    '''
    start_time = time.time()
    vectors = run_embedding(lc_emb, texts)
    duration = time.time() - start_time
    print(f"4️⃣ 문서 임베딩: {type(vectors)}\t{type(vectors[0])}\t{duration:.2f}초")    
    
    '''
    5️⃣ VectorStore 준비/생성 (VectorStore Init / Load Index)
    vs = build_vectorstore(cfg, lc_emb)
    (FAISS면 “인덱스 생성/로드 + persist 경로 준비”에 가까움)
    '''
    start_time = time.time()
    cfg = VectorStoreConfig(
        backend="faiss",
        persist_dir="./vs_faiss",   # FAISS/Chroma는 로컬 저장 경로
        collection="my_docs",       # Chroma/Pinecone에서 주로 사용
        k=5,
        search_type="similarity",
   )
    vs = build_vectorstore(cfg, lc_emb)
    duration = time.time() - start_time
    print(f"5️⃣ VectorStore 준비/생성	{duration:.2f}초")

    '''
    6️⃣ VectorStore 인덱싱/업서트 (Add / Upsert + Index Build)
    ids = vs.add_documents(texts)
    '''
    start_time = time.time()    
    ids = vs.add_documents(texts)
    duration = time.time() - start_time
    print(f"6️⃣ VectorStore 인덱싱/업서트 {type(ids)}	{duration:.2f}초")
    
    
    '''
    7️⃣ Retriever 생성 (Retriever Build)
    retriever = vs.as_retriever(cfg)
    '''
    start_time = time.time()
    retriever = vs.as_retriever(cfg)   
    duration = time.time() - start_time
    print(f"7️⃣ Retriever 생성	{duration:.2f}초")


    query_text = "구글이 얼마 주기로 했는가?"


    # 8️⃣ VectorStore 리트리버 검색
    '''
    8️⃣ RAG 체인 구성
    prompt = PromptTemplate...
    llm = ChatOpenAI...
    chain = {"context": retriever, ...} | prompt | llm | parser
    '''
    start_time = time.time()
    prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved contenxt to answer the question.
            If you don't know the answer, just say that you don't know. Answer in Korean.
            
            #Context: {context}
            #Question: {question}
            #Answer: 
            """
        )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    duration = time.time() - start_time
    print(f"8️⃣ RAG Chain 구성 완료 {duration:.2f}초")
    
    '''
    9️⃣ 체인 실행 (RAG Inference: Retrieve → Prompt → LLM)
    ret = chain.invoke(query_text)
    이 안에서 실제로:
    retriever가 검색 수행
    context가 prompt에 삽입
    LLM 호출
    파싱    
    이 순서로 진행됩니다.
    '''
    start_time = time.time()
    ret = chain.invoke(query_text)
    duration = time.time() - start_time
    print(ret)
    print(f"9️⃣ RAG Chain 실행 완료 {duration:.2f}초")