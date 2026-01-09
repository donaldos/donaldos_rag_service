# 프로젝트 개요: RAG 서비스

## 목적
이 저장소는 PDF 문서를 위한 **Retrieval‑Augmented Generation (RAG)** 파이프라인을 구현합니다. 다음을 보여줍니다:
- PDF 파일 로드
- 의미 있는 청크로 텍스트 분할
- HuggingFace 혹은 OpenAI 모델로 청크 임베딩
- 메모리 내 벡터 스토어에 임베딩 저장
- 사용자 질의에 대해 가장 유사한 청크 검색
- (플레이스홀더) 검색된 청크를 LLM에 전달해 자연어 답변 생성

## 고수준 아키텍처
```
+-------------------+      +----------------------+      +-------------------+
|   문서 로더      | ---> |   청크 레이어        | ---> |   임베딩 레이어   |
+-------------------+      +----------------------+      +-------------------+
                                   |                         |
                                   v                         v
                         +----------------------+   +-------------------+
                         |  벡터 스토어 (FAISS) |   |  SimilarityRanker |
                         +----------------------+   +-------------------+
                                   |                         |
                                   v                         v
                         +----------------------+   +-------------------+
                         |   검색 (search)      |   |   LLM (미구현)    |
                         +----------------------+   +-------------------+
```

## 핵심 패키지 및 모듈
| 패키지 | 모듈 | 주요 클래스/함수 | 설명 |
|--------|------|------------------|------|
| `chunkingclass` | `chunking_*.py` | `CBaseChunkSplitter`, 구체적인 청크 분리기(`CCharTextSplitter`, `CRecursiveCharTextSplitter`, `CTokenTextSplitter`, `CSemanticTextSplitter` 등) | LangChain splitter를 활용한 다양한 텍스트 분할 전략 구현 |
| `embeddingclass` | `embedder_hf.py`, `embedder_openai.py`, `embedder_base.py` | `EmbedderBase`, 구체적인 임베더 구현 | HuggingFace / OpenAI 임베딩 모델을 래핑하고 `embed_documents`·`embed_query` 제공 |
| `vectorstoreclass` | `vs_factory.py` | `VectorStoreFactory` | 임베딩으로부터 FAISS 벡터 스토어를 생성하는 팩터리 |
| `embeddingclass.retrieval` | `retrieval_similarity_ranker.py` | `SimilarityRanker` | 문서 임베딩을 미리 계산하고 dot‑product 기반 유사도 검색 수행 |
| `main_rag.py` | – | `run_chunking`, `run_embedding`, `run_query` | 전체 흐름을 오케스트레이션 |

## 빠른 시작
1. **필요 패키지 설치**
   ```bash
   pip install -r requirements.txt   # langchain, langchain‑community, faiss‑cpu, transformers, python‑dotenv 등 포함
   ```
2. **OpenAI API 키 설정** (OpenAI 임베더 사용 시) `.env` 파일에 추가:
   ```text
   OPENAI_API_KEY=sk-...
   ```
3. **쿼리할 PDF 파일**을 `./data/` 디렉터리에 넣습니다 (예: `SPRI_Report.pdf`).
4. **스크립트 실행**
   ```bash
   python main_rag.py
   ```
   실행 시:
   - PDF 로드
   - `CSemanticTextSplitter` 로 청크화
   - HuggingFace `BAAI/bge-m3` 모델로 청크 임베딩
   - FAISS 벡터 스토어 구축
   - 하드코딩된 질의 `"삼성전자 AI이름은?"`에 대해 유사도 검색 수행
   - 첫 번째 검색 결과 청크를 출력 (플레이스홀더 답변)

## 프로젝트 확장 아이디어
- **실제 LLM 연동**: `run_query` 의 플레이스홀더를 OpenAI `gpt‑4o` 등 LLM 호출로 교체
- **다른 청크 분리기 사용**: 문서 구조에 맞게 `chunkingclass` 의 다른 분리기를 선택
- **벡터 스토어 영속화**: FAISS 인덱스를 디스크에 저장·로드해 시작 시간 단축
- **다중 PDF 지원**: `main_rag.py` 를 확장해 디렉터리 내 모든 PDF 를 처리하고 청크를 합치기

## 디렉터리 구조
```
.
├── chunkingclass/                 # 텍스트 청크 분리기
│   ├── __init__.py
│   ├── chunking_base_splitter.py
│   ├── chunking_char_text_splitter.py
│   ├── chunking_recursive_char_text_splitter.py
│   ├── chunking_token_text_splitter.py
│   ├── chunking_semantic_text_splitter.py
│   └── ...
├── embeddingclass/                # 임베딩 래퍼
│   ├── embedder_base.py
│   ├── embedder_hf.py
│   ├── embedder_openai.py
│   └── retrieval/
│       └── retrieval_similarity_ranker.py
├── vectorstoreclass/              # 벡터 스토어 팩터리
│   └── vs_factory.py
├── data/                          # 샘플 PDF 파일들
├── main_rag.py                    # 엔트리 포인트
├── requirements.txt
└── PROJECT_OVERVIEW_KR.md        # ← 이 파일 (한국어 번역)
```

---
*사용자를 위해 자동으로 생성되었습니다.*
