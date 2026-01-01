import json
from .loaddoc_base import CBaseDocumentLoader
from langchain_core.documents import Document
from typing import List

class CJSONDocumentLoader(CBaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content=json.dumps(data, ensure_ascii=False, indent=2),
                metadata={
                    "source": path,
                    "type": "json",
                }
            )
        ]
