from .embedder_base import EmbedderBase
from .embedder_hf import HuggingFaceEmbedder
from .embedder_openai import OpenAIEmbedder
from .retrieval import SimilarityRanker
from .lc_adapter import LCEmbeddingAdapter

__all__ = [
    "EmbedderBase",
    "OpenAIEmbedder",
    "HuggingFaceEmbedder",
    "SimilarityRanker",
    "LCEmbeddingAdapter",
]
def create_embedder(kind: str, **kwargs: any) -> EmbedderBase:
    """
    Embedder factory

    Args:
        kind: "hf" | "openai"
        **kwargs: 구현체 생성자에 그대로 전달

    Returns:
        EmbedderBase 인터페이스를 만족하는 임베더
    """
    kind = kind.lower().strip()

    if kind in ("hf", "huggingface"):
        return HuggingFaceEmbedder(**kwargs)

    if kind in ("openai",):
        return OpenAIEmbedder(**kwargs)

    raise ValueError(f"Unknown embedder kind: {kind}")