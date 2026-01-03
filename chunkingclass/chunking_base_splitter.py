"""
Author: Donaldos
Date: 2025-12-30

Description: 
텍스트 청킹(Chunking)을 위한 추상 베이스 클래스.

"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

# ABC : Abstract Base Class(추상 베이스 클래스)
class CBaseChunkSplitter(ABC):
    """
    텍스트 청킹(Chunking)을 위한 추상 베이스 클래스.

    이 클래스는 애플리케이션 레벨에서 사용하는 **공통 Chunking 인터페이스**를 정의한다.
    실제 Chunking 방식(문자 단위, 토큰 단위, 문장 단위, 의미 기반 등)은
    본 클래스를 상속받은 하위 클래스에서 구현한다.

    애플리케이션은 구체 구현체에 의존하지 않고,
    오직 CBaseChunkSplitter 인터페이스만을 통해 청킹을 수행한다.
    """

    @abstractmethod
    def create_document(self, contents: list[str]) -> List[Document]:
        """
        원본 텍스트 목록을 입력으로 받아, 청킹된 Document 객체 리스트를 생성한다.

        이 메서드는 RAG 파이프라인에서 **텍스트 분할 → 임베딩 → 벡터 저장소**
        단계로 이어지기 위한 표준 출력 형식을 보장한다.

        구현 클래스는 다음 책임을 가진다:
        - 입력 텍스트를 적절한 기준으로 분할(Chunking)할 것
        - 각 Chunk를 langchain의 Document 객체로 변환할 것
        - 필요한 경우 metadata(출처, chunk index 등)를 포함할 것

        Args:
            contents (list[str]):
                - 청킹 대상이 되는 원본 텍스트 목록
                - 예: 파일 단위 텍스트, 문서 단위 텍스트

        Returns:
            List[Document]:
                - 청킹된 결과물
                - 각 Document는 page_content에 chunk 텍스트를 포함해야 하며,
                  metadata는 선택적으로 포함 가능

        Raises:
            NotImplementedError:
                - 하위 클래스에서 반드시 구현해야 한다.
        """
        ...