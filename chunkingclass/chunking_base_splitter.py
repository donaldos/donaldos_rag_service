# chunkingclass/base_splitter.py
from abc import ABC, abstractmethod
from typing import List

class CBaseChunkSplitter(ABC):

    @abstractmethod
    def create_document(self, contents: str) -> List[str]:
        ...