from langchain_text_splitters import RecursiveCharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CRecursiveCharTextSplitter(CBaseChunkSplitter):

    def __init__(self,chunk_size: int=250, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(            
            separators=["\n\n", "\n", " ", ""],  # 문단 → 문장 → 단어/문자
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            is_separator_regex=False       
        )
    
    def create_document(self,contents: str):
        texts = self.text_splitter.create_documents([contents])
        return texts