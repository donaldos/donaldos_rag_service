from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from .loaddoc_base import CBaseDocumentLoader

class CPDFDocumentLoader(CBaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = PyMuPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "pdf",
            })
        return docs
