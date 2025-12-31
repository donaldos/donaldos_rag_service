import pandas as pd

class ExcelDocumentLoader(BaseDocumentLoader):
    def load(self, path: str) -> List[Document]:
        sheets = pd.read_excel(path, sheet_name=None)
        docs = []

        for sheet_name, df in sheets.items():
            text = df.to_csv(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "type": "xlsx",
                        "sheet": sheet_name,
                    }
                )
            )
        return docs
