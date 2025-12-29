"""
Author: Donaldos
Date: 2025-12-13
Description: 
 일반적으로 텍스트를 분할할대, 글자수나 토큰 수를 기준으로 나누는 방식을 많이 사용하지만 
 sementicchunker는 의미적으로 유사한 문장끼리 묶습니다.
 이러한 접근 방식은 토큰의 개수나 글자수에 영향을 받지 않으므로 생성되는 청크의 크기가 일정하지 않습니다.
 접근방식이 더 상식에 가깝다.
"""
from .chunking_base_splitter import CBaseChunkSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


class CSemanticTextSplitter(CBaseChunkSplitter):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        CSemanticTextSplitter 함수 생성
        Args:
            OpenAIEmbeddings: OpenAIEmbeddings를 이용하여 SemanticChunker를 초기화
            의미단위로 청크를 분할하므로 청크크기나 오버랩을 지정하지 않습니다.

            의미가 유사한 문장을 묶는 역할을 한다.
            먼저 문장간 유사도를 계산한 뒤 각 문장쌍 사이의 거리를 나타냅니다. 그런 다음 계산된 거리값을 그래프 형태로 표현하여 
            문장드 간 거리가 가까운 경우와 먼 경우를 파악한다.
            분할의 기준점의 임계값은 백분위수(percentile), 표준편차(standard_deviation), 사분위수 범위(interquartile)
        """
        # Initialize embeddings and pass them to SemanticChunker with proper parameters
        embeddings = OpenAIEmbeddings()
        self.text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # standard_deviation, interquartile
            breakpoint_threshold_amount=70, # 70th percentile, 70% 이상의 유사도를 가진 문장 간에는 청크를 나누지 않고 함께 묶는다. 그 이후는 분리된다.               
        )
    
    def create_document(self,contents: list[str]):
        # SemanticChunker.create_documents returns List[Document]
        print(type(contents))
        texts = self.text_splitter.create_documents(contents)
        return texts