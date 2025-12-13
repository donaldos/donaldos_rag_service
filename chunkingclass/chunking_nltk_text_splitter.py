import warnings
import nltk
from langchain_text_splitters import NLTKTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CNLTKTextSplitter(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=200, chunk_overlap=0):
        nltk.download('punkt')
        self.text_splitter = NLTKTextSplitter(                
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.split_text(contents)
        return texts