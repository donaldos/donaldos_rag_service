"""
Author: Donaldos
Date: 2025-12-13
Description: 
pip install sentence-transformers
자연어 처리 분야에서 잘 알려진 라이브러리로, SentenceTransformersTokenTextSplitter를 사용하여 텍스트를 분할합니다.
* 시작토큰과 종료 토큰의 개수를 각각 한 개씩 포함하여 총 23개로 설정
텍스트에서 총 토큰 개수를 계산할때 이 두개의 토큰을 제외한 값을 출력
"""

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CSentenceTransformersTokenTextSplitter(CBaseChunkSplitter):

    def __init__(self, chunk_size: int=250, chunk_overlap: int = 0):
        self.text_splitter = SentenceTransformersTokenTextSplitter(            
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        count_start_and_stop_tokens = 2
        text_token_count = self.text_splitter.count_tokens(text=contents) - count_start_and_stop_tokens
        texts = self.text_splitter.create_documents([contents])
        return texts