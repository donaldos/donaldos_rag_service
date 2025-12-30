from langchain_core.embeddings import Embeddings

class LCEmbeddingAdapter(Embeddings):
    def __init__(self, my_embedder):
        self._e = my_embedder

    def embed_documents(self, texts):
        # texts: List[str]
        return self._e.embed_documents(texts)

    def embed_query(self, text):
        # text: str
        return self._e.embed_query(text)
