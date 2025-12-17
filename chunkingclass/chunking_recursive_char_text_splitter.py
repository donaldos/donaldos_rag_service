"""
Author: Donaldos
Date: 2025-12-13
Description: 
 특정문자목록을 기준으로 텍스트를 나누며 기본적으로 단락("\n\n"), 문장("\n"), 단어(" ") 순서로 나누어 재귀적으로 청크를 생성한다.
 chunk_size는 글자수를 기준으로 세어보고 문단 구분이 있는 곳을 찾아서 하나의 청크로 나누겠다는 의미
 chunk_overlap은 청크 사이의 중복되는 글자수를 의미
 구분자 청크사이즈 내에서 발견되지 않을 경우, 청크사이즈를 넘겨서 첨크가 분할되기도 함.
 이 경우 출력시 청크 크기를 넘겼다는 경고 메시지가 나타남.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter
from typing import List
from langchain_core.documents import Document

class CRecursiveCharTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size: int=250, chunk_overlap=50):
        """
        CRecursiveCharTextSplitter 함수 생성
        Args:
            chunk_size (int, optional): chunk size. Defaults to 250.
            chunk_overlap (int, optional): chunk overlap. Defaults to 50.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(            
            separators=["\n\n", "\n", " ", ""],  # 문단 → 문장 → 단어/문자
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            is_separator_regex=False       
        )
    
    def create_document(self,contents: str) -> List[Document]:
        """
        create_document 함수 생성
        Args:
            contents (str): contents
        Returns:
            List[Document]: texts
        """
        texts = self.text_splitter.create_documents([contents])
        return texts