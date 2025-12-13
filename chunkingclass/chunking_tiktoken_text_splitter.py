"""
chunkingclass.TiktokenTextSplitter의 Docstring
pip install tiktoken
Author: Donaldos
Date: 2025-12-13
Description: 
 글자수가 아닌 토큰 수를 기준으로 텍스트를 나누어 청크를 생성한다.
 텍스트를 효율적으로 분할하기 위해 토크나이저를 활용.
 토크나이저는 텍스트를 토큰으로 변환하는데 사용하는 알고리즘
 모델에 입력하기 전에 모델에 입력하기 전에 적절한 토크나이저를 선택하는 것이 중요.
 여기서는 tiktoken을 사용하여 토크나이저를 생성한다.
 tiktoken은 OpenAI에서 만든 빠른 BPE TOKENIZER
""" 
from langchain_text_splitters import CharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CTiktokenTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size: int=250, chunk_overlap:int = 0):
        """
        CTiktokenTextSplitter 함수 생성
        Args:
            chunk_size (int, optional): 토큰 수 기준 청크 크기 설정
            chunk_overlap (int, optional): 청크 사이의 중복 토큰 수 설정
        """
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(            
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts