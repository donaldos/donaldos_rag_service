"""
Author: Donaldos
Date: 2025-12-13
Description: 
 Huggingface(다양한 많은 토크나이저를 제공)의 GPT2TokenizerFast를 사용하여 텍스트를 분할합니다.
 텍스트를 효율적으로 분할하기 위해 토크나이저를 활용.
 토크나이저는 텍스트를 토큰으로 변환하는데 사용하는 알고리즘
 모델에 입력하기 전에 모델에 입력하기 전에 적절한 토크나이저를 선택하는 것이 중요.
 여기서는 GPT2TokenizerFast를 사용하여 토크나이저를 생성한다.
 GPT2TokenizerFast는 OpenAI에서 만든 빠른 BPE TOKENIZER
"""
from transformers import GPT2TokenizerFast
from langchain_text_splitters import CharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CGPT2TokenizerFast(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=300, chunk_overlap=50):    
        """
        CGPT2TokenizerFast 함수 생성
        Args:
            hf_tokenizer (GPT2TokenizerFast): Huggingface tokenizer
            chunk_size (int, optional): chunk size. Defaults to 300.
            chunk_overlap (int, optional): chunk overlap. Defaults to 50.
        """
        self.hf_tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
        self.text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(                
            self.hf_tokenizer,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )

    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts