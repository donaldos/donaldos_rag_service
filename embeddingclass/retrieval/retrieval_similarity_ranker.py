from __future__ import annotations
from typing import List, Dict, Sequence, Union, Optional
from langchain_core.documents import Document
import numpy as np

from embeddingclass.embedder_base import EmbedderBase, DocLike

class SimilarityRanker:
    """
    정규화된 임베딩을 전제로 하면:
      cosine similarity == dot product
    """
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def rank(
        self,
        query: str,
        documents: Sequence[DocLike],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        return_metadata: bool = False,
    ) -> List[Dict]:
        texts: List[str] = [
            d.page_content if isinstance(d, Document) else str(d)
            for d in documents
        ]

        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)        # (d,)
        D = np.asarray(self.embedder.embed_documents(documents), dtype=np.float32)  # (n,d)

        scores = D @ q  # (n,)
        order = np.argsort(-scores)

        results: List[Dict] = []
        for i in order:
            s = float(scores[i])
            if score_threshold is not None and s < score_threshold:
                continue

            item = {"문장": texts[i], "유사도": s}

            if return_metadata and isinstance(documents[i], Document):
                item["metadata"] = documents[i].metadata

            results.append(item)

            if top_k is not None and len(results) >= top_k:
                break

        return results
