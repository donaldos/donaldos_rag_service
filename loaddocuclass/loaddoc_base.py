from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class CBaseDocumentLoader(ABC):
    """
    원문 파일을 List[Document]로 변환하는 공통 인터페이스
    """

    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """
        Args:
            path: 원문 파일 경로
        Returns:
            List[Document]: 추출된 원문 Document 목록
        """
        ...
