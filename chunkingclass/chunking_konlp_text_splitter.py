import warnings
import chunk
from langchain_text_splitters import KonlpyTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CKONLPTextSplitter(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=200, chunk_overlap=50):
        
        self.text_splitter = KonlpyTextSplitter(                
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts