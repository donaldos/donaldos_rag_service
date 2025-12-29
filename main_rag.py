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
from embeddingclass.embedders import EmbedderBase, HuggingFaceEmbedder, OpenAIEmbedder
from embeddingclass.retrieval import SimilarityRanker


from langchain_community.document_loaders import PyMuPDFLoader
#from embeddingclass import CEmbeddingOpenAI
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



if __name__=='__main__':
    load_dotenv()    
    DOCFILEPATH = './data/SPRI_Report.pdf'

    loader = PyMuPDFLoader(DOCFILEPATH)
    docs = loader.load()
    
    splitter = CSemanticTextSplitter(chunk_size=500,chunk_overlap=100)
    texts = run_chunking(splitter,docs)

    #print(texts)

    # 1) OpenAI
    # embedder = OpenAIEmbedder(model="text-embedding-3-small", normalize=True)
    # ranker = SimilarityRanker(embedder)
    # print(ranker.rank(query, texts, top_k=3))

    # 2) HuggingFace (E5)
    hf = HuggingFaceEmbedder(
        #model_name="intfloat/multilingual-e5-large-instruct",
        model_name="BAAI/bge-m3",
        device="mps",
        normalize_embeddings=True,
        use_e5_prefix=True,
    )
    query = "삼성전자 AI는?"
    ranker2 = SimilarityRanker(hf)
    ret_value = ranker2.rank(query, texts, top_k=3)
    print(ret_value)
