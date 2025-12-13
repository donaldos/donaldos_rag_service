from .chunking_base_splitter import CBaseChunkSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


class CSemanticTextSplitter(CBaseChunkSplitter):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        # Initialize embeddings and pass them to SemanticChunker with proper parameters
        embeddings = OpenAIEmbeddings()
        self.text_splitter = SemanticChunker(
            embeddings=embeddings,
            #chunk_size=chunk_size,
            #chunk_overlap=chunk_overlap,
        )
    
    def create_document(self,contents: str):
        # SemanticChunker.create_documents returns List[Document]
        texts = self.text_splitter.create_documents([contents])
        return texts