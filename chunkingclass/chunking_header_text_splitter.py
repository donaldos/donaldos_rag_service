import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from chunkingclass.chunking_base_splitter import CBaseChunkSplitter  # 경로는 프로젝트에 맞게 조정하세요


@dataclass
class HeaderRule:
    """
    헤더를 감지하기 위한 규칙.
    level: 헤더의 계층 레벨(1,2,3...)을 추정하는 함수가 있으면 더 좋지만,
           여기서는 정규식 매칭 그룹/패턴으로 간단히 처리합니다.
    """
    pattern: re.Pattern
    level: int


class CHeaderTextSplitter(CBaseChunkSplitter):
    """
    문서의 '헤더(장/절/제목)' 경계를 기준으로 청킹하는 Splitter.

    - 보고서/백서/매뉴얼 등 '구조가 있는 문서'에 유리
    - metadata에 header_path(상위→하위 헤더)를 유지
    - 섹션 본문이 너무 길면 max_chars 기준으로 추가 분할 가능
    """

    def __init__(
        self,
        max_chars: Optional[int] = 4000,
        overlap_chars: int = 200,
        keep_header_line: bool = True,
        custom_header_rules: Optional[List[HeaderRule]] = None,
    ):
        """
        Args:
            max_chars: 섹션 단위 chunk가 너무 길 경우 추가 분할(문자 기준). None이면 추가 분할 안 함.
            overlap_chars: 추가 분할 시 오버랩 문자 수(0 가능).
            keep_header_line: chunk 본문에 헤더 라인을 포함할지 여부
            custom_header_rules: 프로젝트 문서 스타일에 맞춘 헤더 규칙 추가/교체
        """
        self.max_chars = max_chars
        self.overlap_chars = max(0, overlap_chars)
        self.keep_header_line = keep_header_line

        # 기본 헤더 규칙들(필요 시 프로젝트 문서 스타일에 맞게 조정)
        default_rules = [
            # Markdown headers: #, ##, ### ...
            HeaderRule(re.compile(r"^(#{1,6})\s+(.+)$"), level=1),  # level은 실제론 #개수로 계산
            # Numbered headings: 1. / 1) / 1.1 / 2.3.4
            HeaderRule(re.compile(r"^(\d+(?:\.\d+){0,5})[.)]\s+(.+)$"), level=2),
            # Roman numerals: I. II. III.
            HeaderRule(re.compile(r"^([IVXLCDM]+)[.)]\s+(.+)$", re.IGNORECASE), level=2),
            # Korean report style: 제1장, 제2절, 제3조 등 (조항은 ClauseSplitter에서 더 강하게)
            HeaderRule(re.compile(r"^(제\s*\d+\s*(장|절|항))\s*(.*)$"), level=2),
        ]
        self.rules = custom_header_rules if custom_header_rules else default_rules

    def create_document(self, contents: list[str]) -> List[Document]:
        out: List[Document] = []
        global_chunk_idx = 0

        for source_idx, text in enumerate(contents):
            if not text or not text.strip():
                continue

            sections = self._split_into_sections(text)

            for sec_idx, (header_path, header_line, body) in enumerate(sections):
                chunk_text = self._compose_chunk_text(header_line, body)

                # 섹션 단위가 너무 길면 추가 분할
                if self.max_chars and len(chunk_text) > self.max_chars:
                    sub_chunks = self._split_by_length(chunk_text, self.max_chars, self.overlap_chars)
                    for sub_i, sub in enumerate(sub_chunks):
                        out.append(
                            Document(
                                page_content=sub,
                                metadata={
                                    "source_index": source_idx,
                                    "chunk_index": global_chunk_idx,
                                    "section_index": sec_idx,
                                    "sub_index": sub_i,
                                    "header_path": header_path,
                                    "header_line": header_line,
                                    "splitter": "CHeaderTextSplitter",
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
                                "section_index": sec_idx,
                                "header_path": header_path,
                                "header_line": header_line,
                                "splitter": "CHeaderTextSplitter",
                            },
                        )
                    )
                    global_chunk_idx += 1

        return out

    # -------------------------
    # 내부 구현
    # -------------------------
    def _split_into_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Returns:
            List of (header_path, header_line, body_text)
        """
        lines = text.splitlines()
        header_stack: List[Tuple[int, str]] = []  # (level, title)
        sections: List[Tuple[str, str, str]] = []

        current_header_line = ""
        current_header_path = ""
        buffer: List[str] = []

        def flush():
            nonlocal buffer, current_header_line, current_header_path
            body = "\n".join(buffer).strip()
            if current_header_line or body:
                sections.append((current_header_path, current_header_line, body))
            buffer = []

        for line in lines:
            header_info = self._detect_header(line)
            if header_info is None:
                buffer.append(line)
                continue

            # 새로운 헤더를 만나면 이전 섹션 flush
            flush()

            level, title = header_info
            self._update_header_stack(header_stack, level, title)

            current_header_line = line.strip()
            current_header_path = " > ".join([t for _, t in header_stack])

        # 마지막 flush
        flush()

        # 헤더가 한번도 없던 문서는 header_path 빈 상태로 1개 섹션으로 반환될 수 있음
        return sections

    def _detect_header(self, line: str) -> Optional[Tuple[int, str]]:
        s = line.strip()
        if not s:
            return None

        for rule in self.rules:
            m = rule.pattern.match(s)
            if not m:
                continue

            # Markdown: group(1)=####, group(2)=title
            if rule.pattern.pattern.startswith("^(#{1,6})"):
                hashes = m.group(1)
                title = m.group(2).strip()
                return (len(hashes), title)

            # Numbered: group(1)=1.2.3, group(2)=title
            if "d+(?:\\.d+)" in rule.pattern.pattern:
                num = m.group(1)
                title = m.group(2).strip()
                # 깊이는 점 개수 + 1 정도로 추정
                depth = num.count(".") + 1
                return (depth, f"{num} {title}".strip())

            # Roman numerals
            if "[IVXLCDM]+" in rule.pattern.pattern:
                roman = m.group(1)
                title = m.group(2).strip()
                return (2, f"{roman} {title}".strip())

            # Korean 제n장/절/항
            if "제" in rule.pattern.pattern and "(장|절|항)" in rule.pattern.pattern:
                head = m.group(1).strip()
                tail = (m.group(3) or "").strip()
                title = f"{head} {tail}".strip()
                return (2, title)

            # fallback
            return (rule.level, s)

        return None

    def _update_header_stack(self, stack: List[Tuple[int, str]], level: int, title: str) -> None:
        # 같은 레벨 이상은 pop 후 push
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    def _compose_chunk_text(self, header_line: str, body: str) -> str:
        if self.keep_header_line and header_line:
            if body:
                return f"{header_line}\n{body}".strip()
            return header_line.strip()
        return body.strip()

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
