import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional

from loguru import logger


class SplitStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HYBRID = "hybrid"
    MEDICAL = "medical"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"


@dataclass
class TextChunk:
    id: str
    content: str
    document_id: str
    index: int
    metadata: dict[str, Any]


@dataclass
class SplitStats:
    strategy: str
    num_chunks: int
    total_chars: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    split_time_ms: float


CJK_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "．",
    "！",
    "？",
    "；",
    "；\n",
    ". ",
    "! ",
    "? ",
    "，",
    "、",
    "：",
    " ",
    "",
]


class TextSplitter:

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 75,
        strategy: SplitStrategy = SplitStrategy.HYBRID,
        keep_code_blocks: bool = True,
        keep_headers: bool = True,
        soft_max: Optional[int] = None,
        separators: Optional[List[str]] = None,
        hierarchical_sizes: Optional[List[int]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.keep_code_blocks = keep_code_blocks
        self.keep_headers = keep_headers
        self.soft_max = soft_max or int(chunk_size * 0.75)
        self.separators = separators or CJK_SEPARATORS
        self.hierarchical_sizes = hierarchical_sizes or [1024, 512, 256]
        self._markdown_headers = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def split_text(self, text: str, document_id: str = "") -> List[TextChunk]:
        return self._dispatch(text, document_id)

    def split_text_with_stats(self, text: str, document_id: str = "") -> tuple[List[TextChunk], SplitStats]:
        start = time.perf_counter()
        chunks = self._dispatch(text, document_id)
        elapsed = (time.perf_counter() - start) * 1000

        sizes = [len(c.content) for c in chunks] if chunks else [0]
        stats = SplitStats(
            strategy=self.strategy.value,
            num_chunks=len(chunks),
            total_chars=sum(sizes),
            avg_chunk_size=sum(sizes) / len(sizes) if sizes else 0,
            min_chunk_size=min(sizes) if chunks else 0,
            max_chunk_size=max(sizes) if chunks else 0,
            split_time_ms=round(elapsed, 2),
        )
        return chunks, stats

    def _dispatch(self, text: str, document_id: str) -> List[TextChunk]:
        if self.strategy == SplitStrategy.RECURSIVE:
            return self._split_recursive(text, document_id)
        if self.strategy == SplitStrategy.MARKDOWN:
            return self._split_markdown(text, document_id)
        if self.strategy == SplitStrategy.SENTENCE:
            return self._split_by_sentence(text, document_id)
        if self.strategy == SplitStrategy.HYBRID:
            return self._split_hybrid(text, document_id)
        if self.strategy == SplitStrategy.MEDICAL:
            return self._split_medical(text, document_id)
        if self.strategy == SplitStrategy.SEMANTIC:
            return self._split_semantic(text, document_id)
        if self.strategy == SplitStrategy.HIERARCHICAL:
            return self._split_hierarchical(text, document_id)
        return self._split_by_paragraph(text, document_id)

    # ==================== 递归切分（新增） ====================

    def _split_recursive(self, text: str, document_id: str) -> List[TextChunk]:
        chunks = []
        index = 0
        remaining = text

        for sep in self.separators:
            if sep == "":
                for i in range(0, len(remaining), self.chunk_size - self.chunk_overlap):
                    piece = remaining[i:i + self.chunk_size]
                    if piece.strip():
                        chunks.append(self._create_chunk(piece.strip(), document_id, index, "recursive_char"))
                        index += 1
                remaining = ""
                break

            parts = remaining.split(sep)
            if len(parts) <= 1:
                continue

            merged = []
            current_chunks = []
            current_size = 0

            for part in parts:
                if not part.strip():
                    continue
                part_size = len(part)

                if part_size > self.chunk_size:
                    if current_chunks:
                        merged.append(sep.join(current_chunks))
                        current_chunks = []
                        current_size = 0
                    merged.append(part)
                    continue

                if current_size >= self.soft_max and current_chunks:
                    merged.append(sep.join(current_chunks))
                    current_chunks = self._apply_overlap(current_chunks)
                    current_size = sum(len(c) for c in current_chunks) + len(current_chunks) - 1

                current_chunks.append(part)
                current_size += part_size + len(sep)

            if current_chunks:
                merged.append(sep.join(current_chunks))

            all_ok = True
            for m in merged:
                if len(m) > self.chunk_size:
                    remaining = m
                    all_ok = False
                    break
                if m.strip():
                    chunks.append(self._create_chunk(m.strip(), document_id, index, "recursive"))
                    index += 1

            if all_ok:
                remaining = ""

        if remaining and remaining.strip():
            for i in range(0, len(remaining), self.chunk_size - self.chunk_overlap):
                piece = remaining[i:i + self.chunk_size]
                if piece.strip():
                    chunks.append(self._create_chunk(piece.strip(), document_id, index, "recursive_fallback"))
                    index += 1

        logger.debug(f"Recursive split: {len(chunks)} chunks")
        return chunks

    def _apply_overlap(self, items: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or not items:
            return []
        full = "\n\n".join(items)
        if len(full) <= self.chunk_overlap:
            return [full]
        tail = full[-self.chunk_overlap:]
        return [tail] if tail.strip() else []

    # ==================== Markdown ====================

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

            if current_size >= self.soft_max and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, document_id, index))
                index += 1
                over = self._get_overlap_items(current_chunk)
                current_chunk = over
                current_size = sum(len(x) for x in over)

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
        sentence_pattern = re.compile(r"(?<=[。！？；.!?])\s*|(?<=[。！？；.!?])(?=[\u4e00-\u9fffA-Z])")
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
        return self._split_recursive(text, document_id)

    # ==================== 医疗 ====================

    MEDICAL_BREAKPOINTS = [
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

    def _split_medical(self, text: str, document_id: str) -> List[TextChunk]:
        chunks = []
        index = 0

        pattern = "|".join(self.MEDICAL_BREAKPOINTS)
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
            return self._split_recursive(text, document_id)

        logger.debug(f"Medical split: {len(chunks)} chunks")
        return chunks

    def _split_section(self, section_text: str, document_id: str, start_index: int) -> List[TextChunk]:
        if len(section_text) <= self.chunk_size:
            return [self._create_chunk(section_text, document_id, start_index, "medical_section")]

        paragraphs = re.split(r"\n\s*\n", section_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if para_size > self.chunk_size:
                if current:
                    chunk_text = "\n\n".join(current)
                    chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))
                    current = []
                    current_size = 0

                sub = self._split_large_text(para, document_id, start_index + len(chunks), "medical_subsection")
                chunks.extend(sub)
                continue

            if current_size >= self.soft_max and current:
                chunk_text = "\n\n".join(current)
                chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))
                current = []
                current_size = 0

            current.append(para)
            current_size += para_size

        if current:
            chunk_text = "\n\n".join(current)
            chunks.append(self._create_chunk(chunk_text, document_id, start_index + len(chunks), "medical_section"))

        return chunks

    # ==================== 语义切分（升级） ====================

    _SEMANTIC_MODEL = None
    _SEMANTIC_MODEL_NAME = ""
    _SEMANTIC_MODEL_LOADED = False

    @classmethod
    def _get_semantic_model(cls, model_name: str = "shibing624/text2vec-base-chinese"):
        if not cls._SEMANTIC_MODEL_LOADED or cls._SEMANTIC_MODEL_NAME != model_name:
            try:
                from sentence_transformers import SentenceTransformer
                cls._SEMANTIC_MODEL = SentenceTransformer(model_name)
                cls._SEMANTIC_MODEL_NAME = model_name
                cls._SEMANTIC_MODEL_LOADED = True
            except Exception:
                cls._SEMANTIC_MODEL = None
                cls._SEMANTIC_MODEL_LOADED = False
                raise
        return cls._SEMANTIC_MODEL

    def _split_semantic(self, text: str, document_id: str) -> List[TextChunk]:
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not installed, falling back to recursive split")
            return self._split_recursive(text, document_id)

        try:
            self._get_semantic_model()
            if self._SEMANTIC_MODEL is None:
                logger.warning("semantic model not available, falling back to recursive split")
                return self._split_recursive(text, document_id)
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to recursive split")
            return self._split_recursive(text, document_id)
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}, falling back to recursive")
            return self._split_recursive(text, document_id)

        sentence_pattern = re.compile(r"(?<=[。！？；.!?])\s*|(?<=[。！？；.!?])(?=[\u4e00-\u9fffA-Z])")
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

        if len(sentences) <= 3:
            return [self._create_chunk(text, document_id, 0, "semantic")]

        try:
            model = self._get_semantic_model()
            embeddings = model.encode(sentences)

            similarities = []
            for i in range(len(sentences) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
                )
                similarities.append(sim)

            if not similarities:
                return [self._create_chunk(text, document_id, 0, "semantic")]

            breakpoint_threshold = np.percentile(similarities, 95)
            threshold = np.percentile(similarities, 25)

            windowed_similarities = []
            buffer_size = 1
            for i in range(len(similarities)):
                start_idx = max(0, i - buffer_size)
                end_idx = min(len(similarities), i + buffer_size + 1)
                windowed_similarities.append(np.mean(similarities[start_idx:end_idx]))

            chunks = []
            current_sentences = [sentences[0]]
            current_size = len(sentences[0])
            index = 0

            for i, sim in enumerate(windowed_similarities):
                next_sentence = sentences[i + 1]
                next_size = len(next_sentence)
                is_breakpoint = sim < threshold

                if is_breakpoint or current_size >= self.soft_max:
                    chunk_text = "".join(current_sentences)
                    chunks.append(self._create_chunk(chunk_text, document_id, index, "semantic"))
                    index += 1

                    if self.chunk_overlap > 0 and current_sentences:
                        ol = "".join(current_sentences[-2:]) if len(current_sentences) >= 2 else current_sentences[-1]
                        current_sentences = [ol, next_sentence]
                        current_size = len(ol) + next_size
                    else:
                        current_sentences = [next_sentence]
                        current_size = next_size
                else:
                    current_sentences.append(next_sentence)
                    current_size += next_size

            if current_sentences:
                chunk_text = "".join(current_sentences)
                chunks.append(self._create_chunk(chunk_text, document_id, index, "semantic"))

            logger.debug(f"Semantic split: {len(chunks)} chunks, threshold={threshold:.3f}")
            return chunks

        except Exception as e:
            logger.warning(f"Semantic split failed: {e}, falling back to recursive")
            return self._split_recursive(text, document_id)

    # ==================== 层级切分（可配置） ====================

    def _split_hierarchical(self, text: str, document_id: str) -> List[TextChunk]:
        all_chunks = []

        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        parent_text = []
        parent_index = 0
        parent_target = self.hierarchical_sizes[0]
        child_target = self.hierarchical_sizes[1] if len(self.hierarchical_sizes) > 1 else 512

        for para in paragraphs:
            parent_text.append(para)
            parent_combined = "\n\n".join(parent_text)

            if len(parent_combined) >= parent_target or para == paragraphs[-1]:
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

                    sentences = re.split(r"(?<=[。！？；.!?])\s+", parent_combined)
                    sentences = [s.strip() for s in sentences if s.strip()]

                    child_text = []
                    child_index = parent_index

                    for sentence in sentences:
                        child_text.append(sentence)
                        child_combined = "".join(child_text)

                        if len(child_combined) >= child_target or sentence == sentences[-1]:
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

    # ==================== 合并与工具方法 ====================

    def _merge_into_chunks(self, items: List[str], document_id: str, source_type: str) -> List[TextChunk]:
        chunks = []
        index = 0
        current_items = []
        current_size = 0

        for item in items:
            item_size = len(item)

            if item_size > self.chunk_size:
                if current_items:
                    chunk_text = self._join_items(current_items)
                    chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))
                    index += 1
                    current_items = []
                    current_size = 0

                sub_chunks = self._split_large_text(item, document_id, index, source_type)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                continue

            if current_size >= self.soft_max and current_items:
                chunk_text = self._join_items(current_items)
                chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))
                index += 1
                current_items = self._get_overlap_items(current_items)
                current_size = sum(len(i) for i in current_items)

            current_items.append(item)
            current_size += item_size

        if current_items:
            chunk_text = self._join_items(current_items)
            chunks.append(self._create_chunk(chunk_text, document_id, index, source_type))

        logger.debug(f"Split by {source_type}: {len(chunks)} chunks")
        return chunks

    def _join_items(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return "\n\n".join(items)

    def _get_overlap_items(self, items: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or not items:
            return []
        full = "\n\n".join(items)
        if len(full) <= self.chunk_overlap:
            return [full]
        tail = full[-self.chunk_overlap:]
        return [tail] if tail.strip() else []

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
            chunk_text = text[i:i + self.chunk_size]
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

    def split_text_streaming(self, text: str, document_id: str = "") -> Generator[TextChunk, None, None]:
        text_length = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)
        index = 0

        SENTENCE_BOUNDARIES = "。！？；\n.!?;\r"

        for i in range(0, text_length, step):
            end = min(i + self.chunk_size, text_length)

            if end < text_length:
                search_start = max(i, end - min(100, end - i))
                search_region = text[search_start:end]
                best_pos = -1
                for boundary in SENTENCE_BOUNDARIES:
                    pos = search_region.rfind(boundary)
                    if pos > best_pos:
                        best_pos = pos
                if best_pos >= 0:
                    end = search_start + best_pos + 1

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
    if domain == "medical" and _is_medical_report(text):
        return SplitStrategy.MEDICAL

    if _has_markdown_structure(text):
        return SplitStrategy.MARKDOWN

    if len(text) > 5000 and not _has_clear_structure(text):
        return SplitStrategy.SEMANTIC

    if _has_multi_level_structure(text):
        return SplitStrategy.HIERARCHICAL

    return SplitStrategy.RECURSIVE


def _is_medical_report(text: str) -> bool:
    medical_keywords = [
        "病史摘要", "主诉", "体格检查", "辅助检查", "实验室检查",
        "影像学检查", "诊断", "诊疗计划", "治疗方案", "处置",
        "既往史", "过敏史", "个人史", "家族史", "婚育史",
        "出院小结", "预后", "讨论",
    ]
    return any(kw in text for kw in medical_keywords)


def _has_markdown_structure(text: str) -> bool:
    header_count = len(re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE))
    return header_count >= 3


def _has_clear_structure(text: str) -> bool:
    indicators = [
        r"第[一二三四五六七八九十\d]+[章节部分篇]",
        r"【.+】",
        r"\d+\.\s+[A-Z\u4e00-\u9fff]",
    ]
    return any(re.search(pattern, text) for pattern in indicators)


def _has_multi_level_structure(text: str) -> bool:
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
