from langchain_community.document_loaders import PyMuPDFLoader

class PDFDocumentLoader(BaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = PyMuPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "pdf",
            })
        return docs
