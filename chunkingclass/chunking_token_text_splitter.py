"""
Author: Donaldos
Date: 2025-12-13
Description: 
 토큰 수를 기준으로 텍스트를 나누어 청크를 생성한다.
 토크나이저를 직접 사용하여 텍스트를 바로 토큰 단위로 분할
"""
from langchain_text_splitters import TokenTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CTokenTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size:int=250, chunk_overlap:int=0):
        """
        CTokenTextSplitter 함수 생성
        Args:
            chunk_size (int, optional): 토큰 수 기준 청크 크기 설정
            chunk_overlap (int, optional): 청크 사이의 중복 토큰 수 설정
        """
        self.text_splitter = TokenTextSplitter(            
            chunk_size = 250,
            chunk_overlap = 0,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts