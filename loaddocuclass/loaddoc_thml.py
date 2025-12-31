from langchain_community.document_loaders import BSHTMLLoader

class HTMLDocumentLoader(BaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        loader = BSHTMLLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": path,
                "type": "html",
            })
        return docs
