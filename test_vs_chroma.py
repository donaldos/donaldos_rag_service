from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

# 텍스트 파일을 List[Document]형태로 변환
loader1 = TextLoader("data/input.txt")
loader2 = TextLoader("data/finance.txt")

# 문서분할
splitter_doc1 = loader1.load_and_split(text_splitter)
splitter_doc2 = loader2.load_and_split(text_splitter)

DBPATH = "./vectordb/chroma_db"
db = Chroma.from_documents(documents=splitter_doc1 + splitter_doc2, 
                            embedding=OpenAIEmbeddings(), 
                            persist_directory=DBPATH,
                            collection_name="nlp",
                            )

persis_db = Chroma(persist_directory=DBPATH, 
                    embedding_function=OpenAIEmbeddings(), 
                    collection_name="mydb",
                    )


db2 = Chroma.from_texts(['안녕하세요? 정말 반갑습니다.','제 이름은 무중입니다.'], 
                        embedding=OpenAIEmbeddings(), 
                        persist_directory=DBPATH, 
                        collection_name="mytextdb")

persis_db2 = Chroma(persist_directory=DBPATH, 
                    embedding_function=OpenAIEmbeddings(), 
                    collection_name="mytextdb",
                    )
"""
# 데이터 추가/삭제 관련 테스트
import json
#print(json.dumps(persis_db2.get(), indent=2))

#print(db.similarity_search("TF IDF에 대하여 알려줘",filter={"source": "input.txt"}, k=1))

db.add_documents([Document(page_content="안녕하세요? 정말 반갑습니다.", metadata={"source": "input.txt"},id="1",)])
print(db.get("1"))

db.add_texts(['안녕하세요? 정말 반갑습니다.','제 이름은 무중입니다.'], metadata={"source": "input.txt"},id="1")
print(db.get("1"))
"""

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.25, "fetch_k": 10,"score_threshold": 0.2})
print(retriever.invoke("Word2Vec에 대하여 알려줘"))