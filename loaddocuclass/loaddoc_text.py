class TextDocumentLoader(BaseDocumentLoader):
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
