#from .loaddoc_base import CBaseDocumentLoader
from langchain_core.documents import Document
from .loaddoc_docx import CDocxDocumentLoader
from .loaddoc_pdf import CPDFDocumentLoader
from .loaddoc_text import CTextDocumentLoader
from .loaddoc_xlsx import CExcelDocumentLoader
from .loaddoc_html import CHTMLDocumentLoader
from .loaddoc_json import CJSONDocumentLoader
from pathlib import Path
from typing import List


class CDocumentLoaderFactory:
    """
    원문 파일 포맷에 따라 적절한 DocumentLoader를 선택하여
    공통 출력 포맷인 List[Document]로 변환하는 Factory 클래스.

    이 클래스의 목적은:
    - 파일 확장자 기반으로 로더 선택 로직을 중앙 집중화하고
    - 애플리케이션 및 상위 파이프라인에서
      파일 포맷에 대한 분기(if/else)를 제거하는 것이다.

    Chunking, Embedding, VectorStore 단계에서는
    파일 타입(pdf, docx, xlsx 등)을 전혀 알 필요가 없으며,
    오직 List[Document]만을 입력으로 받도록 설계된다.
    """

    _loaders = {
        ".pdf": CPDFDocumentLoader(),
        ".txt": CTextDocumentLoader(),
        ".docx": CDocxDocumentLoader(),
        ".xlsx": CExcelDocumentLoader(),
        ".html": CHTMLDocumentLoader(),
        ".json": CJSONDocumentLoader(),
        # ".xml": CXMLDocumentLoader(),
    }
    """
    파일 확장자와 DocumentLoader 구현체 간의 매핑 테이블.

    Key:
        - 파일 확장자 (소문자, dot 포함)

    Value:
        - CBaseDocumentLoader를 상속한 로더 인스턴스

    새로운 파일 포맷을 지원하려면:
        1) CBaseDocumentLoader를 상속한 Loader 구현
        2) 해당 Loader를 이 딕셔너리에 등록
    만으로 확장이 가능하다.
    """

    @classmethod
    def load(cls, path: str) -> List[Document]:
        """
        주어진 파일 경로를 기반으로 적절한 DocumentLoader를 선택하여
        원문을 List[Document] 형태로 로드한다.

        이 메서드는 Loader 선택 책임만 가지며,
        실제 파싱/텍스트 추출 로직은 각 Loader 구현체에 위임한다.

        Args:
            path (str):
                - 로드할 원문 파일의 경로
                - 예: "data/manual.pdf", "docs/spec.xlsx"

        Returns:
            List[Document]:
                - 원문에서 추출된 Document 객체 목록
                - 각 Document는 page_content와 metadata를 포함한다

        Raises:
            ValueError:
                - 지원하지 않는 파일 확장자인 경우
        """
        ext = Path(path).suffix.lower()

        if ext not in cls._loaders:
            raise ValueError(f"Unsupported file type: {ext}")

        return cls._loaders[ext].load(path)
