import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from chunkingclass.chunking_base_splitter import CBaseChunkSplitter  # 경로는 프로젝트에 맞게 조정하세요


@dataclass
class ClauseConfig:
    """
    조항 문서 스타일에 따라 정규식을 튜닝할 수 있도록 설정 분리.
    """
    # 제1조 / 제 1 조 / 제1조(제목)
    article_pattern: re.Pattern = re.compile(r"^(제\s*\d+\s*조)\s*(?:\(([^)]+)\))?\s*$")
    # ① ② ③ ... (유니코드 원문자)
    paragraph_pattern: re.Pattern = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)$")
    # 1. / 1) / 1] / 1) 형태
    item_pattern: re.Pattern = re.compile(r"^(\d+)\s*[.)\]]\s*(.*)$")


class CClauseTextSplitter(CBaseChunkSplitter):
    """
    법률/약관/규정의 '조/항/호' 구조를 기준으로 청킹하는 Splitter.

    기본 정책:
    - '제n조'를 최상위 chunk 경계로
    - 옵션으로 '항(①②...)' 또는 '호(1. 2.)' 단위까지 분해 가능
    - metadata에 article/paragraph/item 정보를 명확히 남김
    """

    def __init__(
        self,
        split_level: str = "article",  # "article" | "paragraph" | "item"
        max_chars: Optional[int] = None,
        overlap_chars: int = 0,
        config: Optional[ClauseConfig] = None,
        keep_titles: bool = True,
    ):
        """
        Args:
            split_level:
                - "article": 제n조 단위로만 청킹
                - "paragraph": 제n조 내부를 ①②... 항 단위까지 청킹
                - "item": 제n조/항 내부를 1.2. 호 단위까지 청킹(문서 스타일에 따라 과분절될 수 있음)
            max_chars: 특정 조항이 너무 길 경우 추가 분할(문자 기준). None이면 추가 분할 안 함.
            overlap_chars: 추가 분할 시 오버랩 문자 수
            config: 정규식/스타일 커스터마이징
            keep_titles: (제n조 제목) 라인을 chunk에 포함
        """
        if split_level not in ("article", "paragraph", "item"):
            raise ValueError("split_level must be one of: 'article', 'paragraph', 'item'")

        self.split_level = split_level
        self.max_chars = max_chars
        self.overlap_chars = max(0, overlap_chars)
        self.cfg = config or ClauseConfig()
        self.keep_titles = keep_titles

    def create_document(self, contents: list[str]) -> List[Document]:
        out: List[Document] = []
        global_chunk_idx = 0

        for source_idx, text in enumerate(contents):
            if not text or not text.strip():
                continue

            articles = self._split_articles(text)

            for a_idx, article in enumerate(articles):
                # split_level에 따라 추가 분해
                if self.split_level == "article":
                    candidates = [article]
                elif self.split_level == "paragraph":
                    candidates = self._split_paragraphs(article)
                else:  # "item"
                    candidates = self._split_items(article)

                for local_idx, (meta, chunk_text) in enumerate(candidates):
                    # 너무 길면 추가 분할
                    if self.max_chars and len(chunk_text) > self.max_chars:
                        sub_chunks = self._split_by_length(chunk_text, self.max_chars, self.overlap_chars)
                        for sub_i, sub in enumerate(sub_chunks):
                            out.append(
                                Document(
                                    page_content=sub,
                                    metadata={
                                        "source_index": source_idx,
                                        "chunk_index": global_chunk_idx,
                                        "article_index": a_idx,
                                        "local_index": local_idx,
                                        "sub_index": sub_i,
                                        "splitter": "CClauseTextSplitter",
                                        **meta,
                                    },
                                )
                            )
                            global_chunk_idx += 1
                    else:
                        out.append(
                            Document(
                                page_content=chunk_text,
                                metadata={
                                    "source_index": source_idx,
                                    "chunk_index": global_chunk_idx,
                                    "article_index": a_idx,
                                    "local_index": local_idx,
                                    "splitter": "CClauseTextSplitter",
                                    **meta,
                                },
                            )
                        )
                        global_chunk_idx += 1

        return out

    # -------------------------
    # 내부 구현
    # -------------------------
    def _split_articles(self, text: str) -> List[str]:
        """
        제n조 단위로 분리. '제n조'가 없으면 전체를 하나로 반환.
        """
        lines = text.splitlines()

        articles: List[List[str]] = []
        current: List[str] = []

        def flush():
            nonlocal current
            if current:
                articles.append(current)
            current = []

        for line in lines:
            if self.cfg.article_pattern.match(line.strip()):
                # 새 조항 시작
                flush()
            current.append(line)

        flush()

        if not articles:
            return [text.strip()]

        return ["\n".join(a).strip() for a in articles if "\n".join(a).strip()]

    def _parse_article_header(self, article_text: str) -> Tuple[str, Optional[str]]:
        """
        article_text에서 첫 번째 줄이 제n조 헤더면 (article_id, title) 반환
        """
        first_line = (article_text.splitlines()[0].strip() if article_text.splitlines() else "")
        m = self.cfg.article_pattern.match(first_line)
        if not m:
            return ("", None)
        article_id = re.sub(r"\s+", "", m.group(1))  # "제 1 조" -> "제1조"
        title = (m.group(2) or None)
        return (article_id, title)

    def _split_paragraphs(self, article_text: str) -> List[Tuple[dict, str]]:
        """
        제n조 내부를 ①②... 기준으로 분리.
        항이 없으면 article 단위 그대로 반환.
        """
        article_id, title = self._parse_article_header(article_text)
        lines = article_text.splitlines()

        header_lines: List[str] = []
        body_lines = lines[:]
        if lines:
            # 첫 줄이 제n조 헤더면 헤더로 분리
            if self.cfg.article_pattern.match(lines[0].strip()):
                header_lines = [lines[0]]
                body_lines = lines[1:]

        chunks: List[Tuple[dict, str]] = []
        current: List[str] = []
        current_para = None

        def flush():
            nonlocal current, current_para
            if not current:
                return
            text = "\n".join(current).strip()
            if not text:
                current = []
                return
            meta = {
                "clause_article": article_id,
                "clause_title": title,
                "clause_paragraph": current_para,
                "clause_item": None,
                "clause_id": self._make_clause_id(article_id, current_para, None),
            }
            chunks.append((meta, text))
            current = []

        # 헤더를 각 chunk에 포함시킬지 정책
        prefix = "\n".join(header_lines).strip() if (self.keep_titles and header_lines) else ""

        for line in body_lines:
            s = line.strip()
            m = self.cfg.paragraph_pattern.match(s)
            if m:
                # 새 항 시작
                flush()
                current_para = m.group(1)  # ① 같은 원문자
                current = []
                if prefix:
                    current.append(prefix)
                current.append(s)
            else:
                if not current:
                    # 항이 한 번도 시작되지 않았다면 article 통째로 처리해야 할 수도
                    # 일단 모으되, 나중에 항이 없으면 fallback
                    if prefix:
                        current.append(prefix)
                current.append(line)

        flush()

        # 항(①) 구분이 실제로 없었던 경우: article 그대로 1개
        if all(meta.get("clause_paragraph") is None for meta, _ in chunks):
            text = article_text.strip()
            meta = {
                "clause_article": article_id,
                "clause_title": title,
                "clause_paragraph": None,
                "clause_item": None,
                "clause_id": self._make_clause_id(article_id, None, None),
            }
            return [(meta, text)]

        return chunks

    def _split_items(self, article_text: str) -> List[Tuple[dict, str]]:
        """
        제n조(그리고 가능하면 ①) 내부를 1. 2. '호' 단위로 추가 분리.
        문서 스타일에 따라 과분절될 수 있어 split_level='item'은 선택적으로 사용 권장.
        """
        # 먼저 paragraph 단위로 나눈 다음, 각 paragraph에서 item 분리
        para_chunks = self._split_paragraphs(article_text)
        out: List[Tuple[dict, str]] = []

        for meta, para_text in para_chunks:
            lines = para_text.splitlines()
            current: List[str] = []
            current_item = None

            def flush():
                nonlocal current, current_item
                if not current:
                    return
                text = "\n".join(current).strip()
                if text:
                    new_meta = dict(meta)
                    new_meta["clause_item"] = current_item
                    new_meta["clause_id"] = self._make_clause_id(
                        meta.get("clause_article") or "",
                        meta.get("clause_paragraph"),
                        current_item,
                    )
                    out.append((new_meta, text))
                current = []

            for line in lines:
                s = line.strip()
                m = self.cfg.item_pattern.match(s)
                if m:
                    flush()
                    current_item = m.group(1)  # "1" 같은 숫자
                    current = [s]
                else:
                    current.append(line)

            flush()

            # item 구분이 전혀 없으면 paragraph 그대로 유지
            if not any(m.get("clause_item") is not None for m, _ in out if m.get("clause_id", "").startswith(meta.get("clause_article", ""))):
                # 위 체크는 거칠 수 있으니, 안전하게: para_text 자체도 추가(단, out에 해당 paragraph가 하나도 추가되지 않았다면)
                # 간단히: item이 없으면 out에 paragraph meta로 추가
                if not self._paragraph_has_items(para_text):
                    out.append((meta, para_text.strip()))

        return out

    def _paragraph_has_items(self, text: str) -> bool:
        for line in text.splitlines():
            if self.cfg.item_pattern.match(line.strip()):
                return True
        return False

    def _make_clause_id(self, article_id: str, paragraph: Optional[str], item: Optional[str]) -> str:
        parts = [article_id] if article_id else []
        if paragraph:
            parts.append(paragraph)
        if item:
            parts.append(f"item{item}")
        return "_".join(parts) if parts else "clause"

    def _split_by_length(self, text: str, max_chars: int, overlap: int) -> List[str]:
        if max_chars <= 0:
            return [text]

        chunks: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(0, end - overlap)

        return chunks
