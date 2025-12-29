from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

laoder = PyMuPDFLoader("data/SPRI_Report.pdf")
docs = laoder.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

split_documents = text_splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_documents, embeddings)

retriever = vectorstore.as_retriever()

ret = retriever.invoke("삼성전자가 자체 개발한 AI의 이름은?")

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

question = "삼성전자가 자체 개발한 AI의 이름은?"
ret = chain.invoke(question)
print(ret)