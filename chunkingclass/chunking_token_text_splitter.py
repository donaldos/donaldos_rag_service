from langchain_text_splitters import TokenTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CTokenTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size:int=250, chunk_overlap:int=0):
        self.text_splitter = TokenTextSplitter(            
            chunk_size = 250,
            chunk_overlap = 0,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.split_text(contents)
        return texts