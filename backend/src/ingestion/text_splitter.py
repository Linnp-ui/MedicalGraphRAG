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
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


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
        chunk_size: int = 512,
        chunk_overlap: int = 75,
        strategy: SplitStrategy = SplitStrategy.HYBRID,
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
        elif self.strategy == SplitStrategy.SEMANTIC:
            return self._split_semantic(text, document_id)
        elif self.strategy == SplitStrategy.HIERARCHICAL:
            return self._split_hierarchical(text, document_id)
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

    def _split_semantic(self, text: str, document_id: str) -> List[TextChunk]:
        """语义分块：基于嵌入相似度检测主题边界

        算法流程：
        1. 按句子分割
        2. 对每句生成嵌入（使用轻量级本地模型）
        3. 计算相邻句子余弦相似度
        4. 相似度低于阈值处创建新分块

        Args:
            text: 待分割文本
            document_id: 文档ID

        Returns:
            分割后的文本块列表
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to paragraph split")
            return self._split_by_paragraph(text, document_id)

        sentence_pattern = re.compile(r"(?<=[。.!?！？])\s+|(?<=[。.!?！？])(?=[A-Z\u4e00-\u9fff])")
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

        if len(sentences) <= 3:
            return [self._create_chunk(text, document_id, 0, "semantic")]

        try:
            model = SentenceTransformer("shibing624/text2vec-base-chinese")
            embeddings = model.encode(sentences)

            similarities = []
            for i in range(len(sentences) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
                )
                similarities.append(sim)

            if not similarities:
                return [self._create_chunk(text, document_id, 0, "semantic")]

            threshold = np.percentile(similarities, 25)

            chunks = []
            current_chunk_sentences = [sentences[0]]
            current_size = len(sentences[0])
            index = 0

            for i, sim in enumerate(similarities):
                next_sentence = sentences[i + 1]
                next_size = len(next_sentence)

                if sim < threshold or current_size + next_size > self.chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(self._create_chunk(chunk_text, document_id, index, "semantic"))
                    index += 1

                    if self.chunk_overlap > 0 and current_chunk_sentences:
                        overlap_text = " ".join(current_chunk_sentences[-2:])
                        current_chunk_sentences = [overlap_text, next_sentence]
                        current_size = len(overlap_text) + next_size
                    else:
                        current_chunk_sentences = [next_sentence]
                        current_size = next_size
                else:
                    current_chunk_sentences.append(next_sentence)
                    current_size += next_size

            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(self._create_chunk(chunk_text, document_id, index, "semantic"))

            logger.debug(f"Semantic split: {len(chunks)} chunks, threshold={threshold:.3f}")
            return chunks

        except Exception as e:
            logger.warning(f"Semantic split failed: {e}, falling back to paragraph split")
            return self._split_by_paragraph(text, document_id)

    def _split_hierarchical(self, text: str, document_id: str) -> List[TextChunk]:
        """层级分块：创建父子层级关系

        创建三层结构：
        - Level 1 (Parent): 1024 tokens → 整个章节
        - Level 2 (Child):  512 tokens  → 段落级
        - Level 3 (Detail): 256 tokens  → 句子级

        Args:
            text: 待分割文本
            document_id: 文档ID

        Returns:
            分割后的文本块列表（带层级元数据）
        """
        all_chunks = []

        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        parent_text = []
        parent_index = 0

        for para in paragraphs:
            parent_text.append(para)
            parent_combined = "\n\n".join(parent_text)

            if len(parent_combined) >= 1024 or para == paragraphs[-1]:
                if parent_text:
                    all_chunks.append(
                        self._create_chunk(
                            parent_combined,
                            document_id,
                            parent_index,
                            "hierarchical_parent",
                        )
                    )
                    parent_index += 1

                    sentences = re.split(r"(?<=[。.!?！？])\s+", parent_combined)
                    sentences = [s.strip() for s in sentences if s.strip()]

                    child_text = []
                    child_index = parent_index

                    for sentence in sentences:
                        child_text.append(sentence)
                        child_combined = " ".join(child_text)

                        if len(child_combined) >= 512 or sentence == sentences[-1]:
                            if child_text:
                                all_chunks.append(
                                    self._create_chunk(
                                        child_combined,
                                        document_id,
                                        child_index,
                                        "hierarchical_child",
                                    )
                                )
                                child_index += 1
                                child_text = []

                    parent_text = []

        logger.debug(f"Hierarchical split: {len(all_chunks)} chunks")
        return all_chunks

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


def select_chunking_strategy(text: str, domain: str = "general") -> SplitStrategy:
    """根据文档特征自动选择最优分块策略

    Args:
        text: 文档内容
        domain: 领域类型 ("medical" 或 "general")

    Returns:
        推荐的 SplitStrategy
    """
    if domain == "medical" and _is_medical_report(text):
        return SplitStrategy.MEDICAL

    if _has_markdown_structure(text):
        return SplitStrategy.MARKDOWN

    if len(text) > 5000 and not _has_clear_structure(text):
        return SplitStrategy.SEMANTIC

    if _has_multi_level_structure(text):
        return SplitStrategy.HIERARCHICAL

    return SplitStrategy.HYBRID


def _is_medical_report(text: str) -> bool:
    """检测是否为医疗报告"""
    medical_keywords = [
        "病史摘要", "主诉", "体格检查", "辅助检查", "实验室检查",
        "影像学检查", "诊断", "诊疗计划", "治疗方案", "处置",
        "既往史", "过敏史", "个人史", "家族史", "婚育史",
        "出院小结", "预后", "讨论"
    ]
    return any(kw in text for kw in medical_keywords)


def _has_markdown_structure(text: str) -> bool:
    """检测是否有 Markdown 结构"""
    header_count = len(re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE))
    return header_count >= 3


def _has_clear_structure(text: str) -> bool:
    """检测是否有清晰结构"""
    indicators = [
        r"第[一二三四五六七八九十\d]+[章节部分篇]",
        r"【.+】",
        r"\d+\.\s+[A-Z\u4e00-\u9fff]",
    ]
    return any(re.search(pattern, text) for pattern in indicators)


def _has_multi_level_structure(text: str) -> bool:
    """检测是否有多层结构"""
    h1 = len(re.findall(r"^#\s+", text, re.MULTILINE))
    h2 = len(re.findall(r"^##\s+", text, re.MULTILINE))
    h3 = len(re.findall(r"^###\s+", text, re.MULTILINE))
    return (h1 > 0 and h2 > 0) or (h2 > 0 and h3 > 0)


def split_text(
    text: str,
    document_id: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 75,
    strategy: str = "hybrid",
) -> List[TextChunk]:
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=SplitStrategy(strategy),
    )
    return splitter.split_text(text, document_id)
