from langchain_community.document_loaders import Docx2txtLoader

class DocxDocumentLoader(BaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = Docx2txtLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "docx",
            })
        return docs
