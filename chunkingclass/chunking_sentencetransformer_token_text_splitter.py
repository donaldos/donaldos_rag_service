"""
pip install sentence-transformers
"""

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CSentenceTransformersTokenTextSplitter(CBaseChunkSplitter):

    def __init__(self, chunk_size: int=250, chunk_loverlap: int = 0):
        self.text_splitter = SentenceTransformersTokenTextSplitter(            
            chunk_size = chunk_size,
            chunk_overlap = chunk_loverlap,
        )
    
    def create_document(self,contents: str):
        count_start_and_stop_tokens = 2
        text_token_count = self.text_splitter.count_tokens(text=contents) - count_start_and_stop_tokens
        texts = self.text_splitter.split_text(text=contents)
        return texts