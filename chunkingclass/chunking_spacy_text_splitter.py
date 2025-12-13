"""
Author: Donaldos
Date: 2025-12-13
Description: 
 고급 자연어 처리를 위한 오픈소스 소프트웨어 라이브러리,
 내부적으로 spacy tokenizer를 사용하여 텍스트를 분할
 텍스트를 분할 할때, spaCy의 토크나이저를 활용하여 단어와 문장을 이해하며,
 설정된 청크 크기와 중복을 기반으로 텍스트를 나눕니다.

 spaCy를 사용하려면 먼저 라이브러리르 설치하고 필요한 언어모델을 다운로드해야합니다.
 pip install -qU spacy
 python -m spacy download ko_core_news_sm(en_core_web_sm)
"""
import warnings
from langchain_text_splitters import SpacyTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CSpacyTextSplitter(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=200, chunk_overlap=50):
        """
        CSpacyTextSplitter 함수 생성
        Args:
            pipeline (str, optional): pipeline. Defaults to "ko_core_news_sm".
            chunk_size (int, optional): chunk size. Defaults to 200.
            chunk_overlap (int, optional): chunk overlap. Defaults to 50.
        """
        self.text_splitter = SpacyTextSplitter(    
            pipeline="ko_core_news_sm",        
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts