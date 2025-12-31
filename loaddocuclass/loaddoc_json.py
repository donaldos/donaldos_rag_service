import json

class JSONDocumentLoader(BaseDocumentLoader):
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
