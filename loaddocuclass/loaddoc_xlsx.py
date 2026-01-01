import pandas as pd
from .loaddoc_base import CBaseDocumentLoader
from typing import List
from langchain_core.documents import Document   
class CExcelDocumentLoader(CBaseDocumentLoader):
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
