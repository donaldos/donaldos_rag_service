from langchain_community.document_loaders import BSHTMLLoader
from .loaddoc_base import CBaseDocumentLoader
from typing import List
from langchain_core.documents import Document   

class CHTMLDocumentLoader(CBaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = BSHTMLLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "html",
            })
        return docs
