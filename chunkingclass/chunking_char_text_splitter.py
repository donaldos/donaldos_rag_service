"""
Author: Donaldos
Date: 2025-12-13
Description: 
 기본적으로 "\n\n"를 구분자로 기준으로 글자단위로 텍스트를 분할.
 chunk_size는 글자수를 기준으로 세어보고 문단 구분이 있는 곳을 찾아서 하나의 청크로 나누겠다는 의미
 chunk_overlap은 청크 사이의 중복되는 글자수를 의미
 구분자 청크사이즈 내에서 발견되지 않을 경우, 청크사이즈를 넘겨서 첨크가 분할되기도 함.
 이 경우 출력시 청크 크기를 넘겼다는 경고 메시지가 나타남.
"""

from langchain_text_splitters import CharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter
from langchain_core.documents import Document
from typing import List

class CCharTextSplitter(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=210, chunk_overlap: int=0):
        """
        CCharTextSplitter 함수 생성
        Args:
            chunk_size (int, optional): chunk size. Defaults to 210.
            chunk_overlap (int, optional): chunk overlap. Defaults to 0.
        """
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",                           # 텍스트를 분할 할 구분자
            chunk_size = chunk_size,                    # 청크의 최대 크기(문자수)
            chunk_overlap = chunk_overlap,              # 청크간의 중복되는 문자수
            length_function=len,                        # 텍스트의 길이를 계산
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