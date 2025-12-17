"""
Author: Donaldos
Date: 2025-12-17
Description: Embedder Base Class
 - 임베딩 제공자 공용 인터페이스
 - 임베딩 제공자를 교체 가능하게 만들기 위한 추상화
"""
from __future__ import annotations
from typing import Protocol, List, Sequence, Union
from langchain_core.documents import Document

"""
도메인 추상화: 임ㅔ딩 입력은 문자열일 수도 있고, Documents일 수도 있다.

“문서”라는 개념을 구체 타입으로 고정하지 않겠다는 선언
임베딩 관점에서는 
1) "안녕하세요" 같은 순수 텍스트와
2)Document(page_content=..., metadata=...) 같은 구조화 문서
둘 다 임베딩 대상이라는 공통 개념으로 취급
"""
DocLike = Union[str, Document]

"""
추상 클래스(ABC) 대신 Protocol을 쓴 이유는

"""

class Embedder(Protocol):
    """
    행동 기반 추상화: 이 두 메서드를 제공할 수 있다면 그 객체는 Embedder   
    Protocol = “상속 없는 인터페이스” -> Java / C++의 interface와 개념적으로 동일
    """
    def embed_query(self, text: str) -> List[float]: ...                              #  → vector

    # Sequence의미: list, tuple, generator 모두 허용
    def embed_documents(self, documents: Sequence[DocLike]) -> List[List[float]]: ... #  → vectors