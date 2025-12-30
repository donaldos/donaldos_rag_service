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
from embeddingclass.embedder_base import EmbedderBase, DocLike
from embeddingclass import create_embedder
from embeddingclass.lc_adapter import LCEmbeddingAdapter

#from embeddingclass.embedder_hf import HuggingFaceEmbedder
#from embeddingclass.embedder_openai import OpenAIEmbedder
from embeddingclass.retrieval import SimilarityRanker

from vectorstoreclass import VectorStoreConfig, build_vectorstore
from vectorstoreclass.util import attach_doc_and_chunk_ids, build_where_filter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from typing import Sequence, Union

DocLike = Union[str, Document]

def run_chunking(splitter: CBaseChunkSplitter, docs: Sequence[DocLike]):
    
    if docs and isinstance(docs[0], Document):
        texts = [doc.page_content for doc in docs]
    else:
        texts = docs

    texts = splitter.create_document(texts)

    return texts

def run_embedding(embedder: EmbedderBase, texts: Sequence[DocLike]):
    v = embedder.embed_documents(texts)
    
    return v

def run_query(embedder: EmbedderBase, query: str, ranker: SimilarityRanker):
    # 1) 질문을 임베딩
    query_vec = embedder.embed_query(query)
    # 2) 가장 유사한 문서 검색
    docs = ranker.rank(query_vec, texts, top_k=3)
    #docs = ranker.search(query_vec, top_k=3)   # Document 리스트 반환
    
    # 3) LLM에게 질문 + 검색된 문서 전달 (예시)
    # 여기서는 실제 LLM 구현이 없으니 임시로 첫 번째 문서 내용을 반환
    # 실제 환경에서는 `llm.generate_answer(query, docs)` 와 같은 호출을 사용
    answer = docs[0].page_content if docs else "No relevant document found."
    return answer


if __name__=='__main__':
   load_dotenv()
   DOCFILEPATH = './data/SPRI_Report.pdf'

   # 1️⃣ 문서 로드: docs: List[Document] (PDF 로드)
   loader = PyMuPDFLoader(DOCFILEPATH)
   docs = loader.load()                     # -> List[Document]
   print(f"1️⃣ 문서 로드: {type(docs)}\t{type(docs[0])}")
   
   # 2️⃣ 텍스트 청킹: texts: List[Document] (청킹 결과)
   splitter = CSemanticTextSplitter(chunk_size=500, chunk_overlap=100)
   texts = run_chunking(splitter, docs)     # List[Document] 반환
   print(f"2️⃣ 텍스트 청킹: {type(texts)}\t{type(texts[0])}")
   
   # 3️⃣ 임베더 생성 (예: HuggingFace)
   embedder = create_embedder(
        "hf",
        model_name="BAAI/bge-m3",
        device="mps",
        normalize_embeddings=True,
        use_e5_prefix=True,
   )
   lc_emb = LCEmbeddingAdapter(embedder)
   print(f"3️⃣ 임베더 생성 완료")

   # 4️⃣ 임베딩 및 랭커 초기화   
   #ranker = SimilarityRanker(embedder)   # 필요 시 추가 옵션 전달
   #print(f"4️⃣ 임베딩 및 랭커 초기화 완료")

   #ranked_results = ranker.rank("삼성전자 AI이름은?", texts, top_k=3)
   #for result in ranked_results:
   #    print(result)
   #    print("--------------------------")

   # 5️⃣ 질문 → 답변
   #answer = run_query(embedder, "삼성전자 AI이름은?", ranker)
   #print("Answer:", answer)

    # ----------------------------
    # 3) VectorStore 생성/로드
    # ----------------------------
    # backend 교체: "faiss" | "chroma" | "pinecone"
    # 

   query_text = "구글이 얼마 주기로 했는가?"
   cfg = VectorStoreConfig(
        backend="faiss",
        persist_dir="./vs_faiss",   # FAISS/Chroma는 로컬 저장 경로
        collection="my_docs",       # Chroma/Pinecone에서 주로 사용
        k=5,
        search_type="similarity",
   )
   vs = build_vectorstore(cfg, lc_emb)
   print(f"5️⃣ VectorStore 생성 완료")

   ids = vs.add_documents(texts)
   print(f"6️⃣ VectorStore 인덱싱 완료 {type(ids)}")
   
   retriever = vs.as_retriever(cfg)   
   
   
   results = retriever.invoke(query_text)
   
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

   ret = chain.invoke(query_text)
   print(ret)

"""
   print("\n=== Top Results ===")
   for i, d in enumerate(results, start=1):
       score_info = ""  # retriever에 따라 score가 metadata로 안 내려올 수도 있음
       print(f"\n[{i}] source={d.metadata.get('source')} doc_id={d.metadata.get('doc_id')} chunk_id={d.metadata.get('chunk_id')}{score_info}")
       print(d.page_content[:500])

"""