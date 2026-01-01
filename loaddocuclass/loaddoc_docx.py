from langchain_community.document_loaders import Docx2txtLoader
from typing import List
from .loaddoc_base import CBaseDocumentLoader
from langchain_core.documents import Document   

class CDocxDocumentLoader(CBaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = Docx2txtLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "docx",
            })
        return docs
