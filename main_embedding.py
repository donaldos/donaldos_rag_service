from embeddingclass.embedders.embedder_openai import OpenAIEmbedder
from embeddingclass.embedders.embedder_hf import HuggingFaceEmbedder
from embeddingclass.retrieval.retrieval_similarity_ranker import SimilarityRanker
from dotenv import load_dotenv

load_dotenv()


texts = [
    "안녕하세요? 만나서 반가워요.",
    "LangChain은 LLM 애플리케이션 개발을 단순화하는 프레임워크입니다.",
    "RAG는 외부 지식을 활용해 응답 품질을 향상시키는 기법입니다.",
]

query = "랭체인이 무엇인지 알려줘"

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
ranker2 = SimilarityRanker(hf)
print(ranker2.rank(query, texts, top_k=3))
