# splitter_registry.py
from typing import Type
from chunkingclass import CBaseChunkSplitter
"""
다양한 Chunk Splitter 구현체들
공통점:
 - 모두 CBaseChunkSplitter 인터페이스를 따름
현재 실제로 쓰는 것은:
 - CSemanticTextSplitter
→ 전략 패턴 (Strategy Pattern)
"""
from chunkingclass import ( 
    CBaseChunkSplitter, 
    CCharTextSplitter, 
    CRecursiveCharTextSplitter, 
    CTiktokenTextSplitter, 
    CTokenTextSplitter, 
    CSpacyTextSplitter, 
    CSentenceTransformersTokenTextSplitter,
    CNLTKTextSplitter,                          # Not completed
    CKONLPTextSplitter,                         # Not completed     
    CGPT2TokenizerFast,                         # Not completed
    CSemanticTextSplitter,
    CClauseTextSplitter,
    CHeaderTextSplitter,
)

SPLITTER_REGISTRY: dict[str, Type[CBaseChunkSplitter]] = {
    "char": CCharTextSplitter,
    "recursive_char": CRecursiveCharTextSplitter,
    "tiktoken": CTiktokenTextSplitter,
    "token": CTokenTextSplitter,
    "spacy": CSpacyTextSplitter,
    "sentence_transformers": CSentenceTransformersTokenTextSplitter,
    "semantic": CSemanticTextSplitter,
    "clause": CClauseTextSplitter,
    "header": CHeaderTextSplitter,
}

def create_splitter(
    splitter_type: str,
    **kwargs
) -> CBaseChunkSplitter:
    try:
        splitter_cls = SPLITTER_REGISTRY[splitter_type]
    except KeyError:
        raise ValueError(f"Unsupported splitter type: {splitter_type}")

    return splitter_cls(**kwargs)