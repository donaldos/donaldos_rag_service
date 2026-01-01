from .loaddoc_base import CBaseDocumentLoader
from typing import List
from langchain_core.documents import Document   

class CTextDocumentLoader(CBaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": path,
                    "type": "text",
                }
            )
        ]
