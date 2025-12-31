from pathlib import Path

class DocumentLoaderFactory:
    _loaders = {
        ".pdf": PDFDocumentLoader(),
        ".txt": TextDocumentLoader(),
        ".docx": DocxDocumentLoader(),
        ".xlsx": ExcelDocumentLoader(),
        ".html": HTMLDocumentLoader(),
        ".json": JSONDocumentLoader(),
        # ".xml": XMLDocumentLoader(),
    }

    @classmethod
    def load(cls, path: str) -> List[Document]:
        ext = Path(path).suffix.lower()
        if ext not in cls._loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._loaders[ext].load(path)
