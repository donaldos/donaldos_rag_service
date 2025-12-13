"""
chunkingclass.TiktokenTextSplitter의 Docstring
pip install tiktoken
"""
from langchain_text_splitters import CharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CTiktokenTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size: int=250, chunk_overlap:int = 0):
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(            
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts