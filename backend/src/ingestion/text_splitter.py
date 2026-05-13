import re
from typing import Any, Generator, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class SplitStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HYBRID = "hybrid"
    MEDICAL = "medical"


@dataclass
class TextChunk:
    id: str
    content: str
    document_id: str
    index: int
    metadata: dict[str, Any]


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: SplitStrategy = SplitStrategy.PARAGRAPH,
        keep_code_blocks: bool = True,
        keep_headers: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.keep_code_blocks = keep_code_blocks
        self.keep_headers = keep_headers
        self._markdown_headers = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def split_text(self, text: str, document_id: str = "") -> List[TextChunk]:
        if self.strategy == SplitStrategy.MARKDOWN:
            return self._split_markdown(text, document_id)
        elif self.strategy == SplitStrategy.SENTENCE:
            return self._split_by_sentence(text, document_id)
        elif self.strategy == SplitStrategy.HYBRID:
            return self._split_hybrid(text, document_id)
        elif self.strategy == SplitStrategy.MEDICAL:
            return self._split_medical(text, document_id)
        else:
            return self._split_by_paragraph(text, document_id)

    def _split_markdown(self, text: str, document_id: str) -> List[TextChunk]:
        chunks = []
        index = 0
        current_chunk = []
        current_size = 0

        lines = text.split("\n")
        sections = self._parse_markdown_sections(lines)

        for section in sections:
            section_size = len(section["content"])
            header = section.get("header", "")

            if section_size > self.chunk_size:
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, document_id, index, header))
                    index += 1
                    current_chunk = []
                    current_size = 0

                sub_chunks = self._split_large_text(section["content"], document_id, index, header)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                continue

            if current_size + section_size > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, document_id, index))
                index += 1
                current_chunk = []
                current_size = 0

            if header and self.keep_headers:
                current_chunk.append(header)
                current_size += len(header) + 1

            if section["content"]:
                current_chunk.append(section["content"])
                current_size += section_size

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, document_id, index))

        logger.debug(f"Markdown split: {len(chunks)} chunks")
        return chunks

    def _parse_markdown_sections(self, lines: List[str]) -> List[dict]:
        sections = []
        current_section = {"header": "", "content": ""}
        in_code_block = False

        for line in lines:
            code_start = re.match(r"^```", line)
            if code_start:
                in_code_block = not in_code_block
                current_section["content"] += line + "\n"
                continue

            if in_code_block:
                current_section["content"] += line + "\n"
                continue

            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"header": line, "content": ""}
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _split_by_sentence(self, text: str, document_id: str) -> List[TextChunk]:
        sentence_pattern = re.compile(r"(?<=[。.!?！？])\s+|(?<=[。.!?！？])(?=[A-Z\u4e00-\u9fff])")
        sentences = sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return self._merge_into_chunks(sentences, document_id, "sentence")

    def _split_by_paragraph(self, text: str, document_id: str) -> List[TextChunk]:
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return self._merge_into_chunks(paragraphs, document_id, "paragraph")

    def _split_hybrid(self, text: str, document_id: str) -> List[TextChunk]:
        if "#" in text or "```" in text:
            return self._split_markdown(text, document_id)
        return self._split_by_sentence(text, document_id)

    def _split_medical(self, text: str, document_id: str) -> List[TextChunk]:
        """医疗文档专用分割，识别医疗报告结构

        医疗文档通常有固定的章节结构：
        - 病史摘要/主诉
        - 体格检查
        - 辅助检查/实验室检查
        - 诊断
        - 治疗方案/处置

        Args:
            text: 待分割文本
            document_id: 文档ID

        Returns:
            分割后的文本块列表
        """
        chunks = []
        index = 0

        medical_breakpoints = [
            r"【?\s*病史摘要\s*】?",
            r"【?\s*主诉\s*】?",
            r"【?\s*体格检查\s*】?",
            r"【?\s*辅助检查\s*】?",
            r"【?\s*实验室检查\s*】?",
            r"【?\s*影像学检查\s*】?",
            r"【?\s*诊断\s*】?",
            r"【?\s*诊疗计划\s*】?",
            r"【?\s*治疗方案\s*】?",
            r"【?\s*处置\s*】?",
            r"【?\s*讨论\s*】?",
            r"【?\s*预后\s*】?",
            r"【?\s*出院小结\s*】?",
            r"【?\s*既往史\s*】?",
            r"【?\s*过敏史\s*】?",
            r"【?\s*个人史\s*】?",
            r"【?\s*家族史\s*】?",
            r"【?\s*婚育史\s*】?",
        ]

        pattern = "|".join(medical_breakpoints)
        parts = re.split(f"({pattern})", text)

        current_section = ""
        for part in parts:
            if re.match(f"^({pattern})$", part.strip()):
                if current_section.strip():
                    section_chunks = self._split_section(current_section.strip(), document_id, index)
                    chunks.extend(section_chunks)
                    index += len(section_chunks)
                current_section = part
            else:
                current_section += part

        if current_section.strip():
            section_chunks = self._split_section(current_section.strip(), document_id, index)
            chunks.extend(section_chunks)

        if not chunks:
            return self._split_by_paragraph(text, document_id)

        logger.debug(f"Medical split: {len(chunks)} chunks from {len(parts)} parts")
        return chunks

    def _split_section(self, section_text: str, document_id: str, start_index: int) -> List[TextChunk]:
        """分割医疗章节，保持语义完整

        Args:
            section_text: 章节文本
            document_id: 文档ID
            start_index: 起始索引

        Returns:
            分割后的文本块列表
        """
        if len(section_text) <= self.chunk_size:
            return [self._create_chunk(section_text, document_id, start_index, "medical_section")]

        paragraphs = re.split(r"\n\s*\n", section_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunks = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if para_size > self.chunk_size:
                if current_chunks:
                    chunk_text = "\n\n".join(current_chunks)
                    chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))
                    current_chunks = []
                    current_size = 0

                sub_chunks = self._split_large_text(para, document_id, start_index + len(chunks), "medical_subsection")
                chunks.extend(sub_chunks)
                continue

            if current_size + para_size > self.chunk_size and current_chunks:
                chunk_text = "\n\n".join(current_chunks)
                chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))
                current_chunks = []
                current_size = 0

            current_chunks.append(para)
            current_size += para_size

        if current_chunks:
            chunk_text = "\n\n".join(current_chunks)
            chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))

        return chunks

    def _merge_into_chunks(
        self, items: List[str], document_id: str, source_type: str
    ) -> List[TextChunk]:
        chunks = []
        index = 0
        current_chunks = []
        current_size = 0

        for item in items:
            item_size = len(item)

            if item_size > self.chunk_size:
                if current_chunks:
                    chunk_text = self._join_items(current_chunks)
                    chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))
                    index += 1
                    current_chunks = []
                    current_size = 0

                sub_chunks = self._split_large_text(item, document_id, index, source_type)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                continue

            if current_size + item_size > self.chunk_size and current_chunks:
                chunk_text = self._join_items(current_chunks)
                chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))
                index += 1

                if self.chunk_overlap > 0 and current_chunks:
                    overlap = self._get_overlap(current_chunks)
                    current_chunks = [overlap] if overlap else []
                    current_size = len(overlap) if overlap else 0
                else:
                    current_chunks = []
                    current_size = 0

            current_chunks.append(item)
            current_size += item_size

        if current_chunks:
            chunk_text = self._join_items(current_chunks)
            chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))

        logger.debug(f"Split by {source_type}: {len(chunks)} chunks")
        return chunks

    def _join_items(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return "\n\n".join(items)

    def _get_overlap(self, items: List[str]) -> str:
        if not items or self.chunk_overlap <= 0:
            return ""
        full_text = "\n\n".join(items)
        if len(full_text) <= self.chunk_overlap:
            return full_text
        return full_text[-self.chunk_overlap :]

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        index: int,
        source: Optional[str] = None,
    ) -> TextChunk:
        return TextChunk(
            id=f"{document_id}_chunk_{index}",
            content=content,
            document_id=document_id,
            index=index,
            metadata={"source": source or "regular", "strategy": self.strategy.value},
        )

    def _split_large_text(
        self, text: str, document_id: str, start_index: int, source: str = "large_text"
    ) -> List[TextChunk]:
        chunks = []
        text_length = len(text)
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, text_length, step):
            chunk_text = text[i : i + self.chunk_size]
            chunks.append(
                TextChunk(
                    id=f"{document_id}_chunk_{start_index + i}",
                    content=chunk_text,
                    document_id=document_id,
                    index=start_index + i,
                    metadata={"source": source, "strategy": self.strategy.value},
                )
            )

        return chunks

    def split_text_streaming(
        self, text: str, document_id: str = ""
    ) -> Generator[TextChunk, None, None]:
        """流式文本分割，降低内存占用

        适用于超长文档处理，避免一次性加载所有chunk到内存

        Args:
            text: 待分割文本
            document_id: 文档ID

        Yields:
            TextChunk: 分割后的文本块
        """
        text_length = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)
        index = 0

        for i in range(0, text_length, step):
            end = min(i + self.chunk_size, text_length)

            if end < text_length:
                for j in range(min(50, end - i), 0, -1):
                    boundary_char = text[end - j - 1] if end - j - 1 >= 0 else ''
                    if boundary_char in "。！？.\n\t,，、；：":
                        end = end - j
                        break

            chunk_text = text[i:end].strip()
            if not chunk_text:
                continue

            yield TextChunk(
                id=f"{document_id}_chunk_{i}",
                content=chunk_text,
                document_id=document_id,
                index=index,
                metadata={"source": "streaming", "strategy": self.strategy.value},
            )
            index += 1


def split_text(
    text: str,
    document_id: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "paragraph",
) -> List[TextChunk]:
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=SplitStrategy(strategy),
    )
    return splitter.split_text(text, document_id)
