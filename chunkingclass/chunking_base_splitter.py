# chunkingclass/base_splitter.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class CBaseChunkSplitter(ABC):

    @abstractmethod
    def create_document(self, contents: str) -> List[Document]:
        ...