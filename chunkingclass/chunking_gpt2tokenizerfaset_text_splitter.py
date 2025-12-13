from transformers import GPT2TokenizerFast
from langchain_text_splitters import CharacterTextSplitter
from .chunking_base_splitter import CBaseChunkSplitter

class CGPT2TokenizerFast(CBaseChunkSplitter):
    def __init__(self,chunk_size: int=300, chunk_overlap=50):    
        self.hf_tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
        self.text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(                
            self.hf_tokenizer,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )

    def create_document(self,contents: str):
        texts = self.text_splitter.split_text(contents)
        return texts