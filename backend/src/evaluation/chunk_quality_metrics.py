"""切分质量评估指标体系

用于量化评估文本切分策略的质量，每个指标定义明确的计算方式和通过阈值。
所有指标计算 <= 1ms，不依赖外部模型。

指标总览：
  - consistency:    语义完整性（chunk 内句子主题一致度）
  - boundary:       边界质量（在自然边界切分占比）
  - density:        信息密度（非噪音字符占比）
  - overlap_eff:    重叠利用率（overlap 区域在相邻 chunk 的复用度）
  - length_stab:    长度稳定性（chunk 大小变异系数）
  - retrieval_recall: 检索召回（核心术语在单个 chunk 中完整出现率）
"""
import re
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkQualityReport:
    strategy_name: str
    num_chunks: int
    total_chars: int

    consistency: float       # [0, 1]  阈值 >= 0.70
    boundary: float          # [0, 1]  阈值 >= 0.85
    density: float           # [0, 1]  阈值 >= 0.60
    overlap_eff: float       # [0, 1]  阈值 >= 0.30
    length_stab: float       # >= 0    阈值 <= 0.50
    retrieval_recall: float  # [0, 1]  阈值 >= 0.80

    overlap_ratio: float
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    split_time_ms: float
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(not v for v in self.violations)

    @property
    def overall_score(self) -> float:
        return (
            self.consistency * 0.25
            + self.boundary * 0.20
            + self.density * 0.10
            + self.overlap_eff * 0.10
            + (1 - min(self.length_stab, 1)) * 0.10
            + self.retrieval_recall * 0.25
        )

    def summary(self) -> str:
        return (
            f"[{self.strategy_name}] "
            f"score={self.overall_score:.3f} "
            f"chunks={self.num_chunks} "
            f"consist={self.consistency:.2f} "
            f"bound={self.boundary:.2f} "
            f"recall={self.retrieval_recall:.2f} "
            f"time={self.split_time_ms:.1f}ms"
        )


class ChunkQualityEvaluator:
    """切分质量评估器 - 所有指标纯计算无模型依赖"""

    # 通过阈值
    THRESHOLDS = {
        "consistency": (0.70, "语义完整性低于 0.70"),
        "boundary": (0.85, "边界质量低于 0.85"),
        "density": (0.60, "信息密度低于 0.60"),
        "overlap_eff": (0.30, "重叠利用率低于 0.30"),
        "length_stab": (0.50, "长度稳定性超过 0.50"),
        "retrieval_recall": (0.80, "检索召回低于 0.80"),
    }

    # 自然边界正则（句号、段落、标题）
    _BOUNDARY_PATTERN = re.compile(r"[。！？；\n]$|^#{1,6}\s")
    # 噪音字符
    _NOISE_PATTERN = re.compile(r"\s{3,}|[\t\r]+|(.)\1{5,}")
    # 核心术语（中文医学术语 + 英文术语模式）
    _TERM_PATTERN = re.compile(r"[A-Za-z_]{3,}|[\u4e00-\u9fff]{2,}")
    # 句子边界
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])(?=[\u4e00-\u9fffA-Z])")
    # 段落边界
    _PARA_BOUNDARY = re.compile(r"\n\s*\n")

    def evaluate(
        self,
        chunks: List[str],
        original_text: str,
        strategy_name: str = "unknown",
        split_time_ms: float = 0.0,
        overlap_size: int = 0,
    ) -> ChunkQualityReport:
        if not chunks:
            return ChunkQualityReport(
                strategy_name=strategy_name,
                num_chunks=0, total_chars=0,
                consistency=0.0, boundary=0.0, density=0.0,
                overlap_eff=0.0, length_stab=0.0, retrieval_recall=0.0,
                overlap_ratio=0.0, avg_chunk_size=0.0,
                min_chunk_size=0, max_chunk_size=0, split_time_ms=split_time_ms,
                violations=["空结果"],
            )

        num = len(chunks)
        sizes = [len(c) for c in chunks]
        total = sum(sizes)
        avg = total / num
        min_s = min(sizes)
        max_s = max(sizes)
        overlap_ratio = overlap_size / avg if avg > 0 else 0

        # --- 1. 语义完整性 (consistency) ---
        # 方法：统计每个 chunk 中非 ending 边界的句子占比
        # 若 chunk 在句号处切分，说明保持了完整句义
        consist_scores = []
        for c in chunks:
            c = c.strip()
            if not c:
                consist_scores.append(1.0)
                continue
            # 判断 chunk 末尾是否结束在自然边界
            ends_natural = bool(self._BOUNDARY_PATTERN.search(c[-1:]))
            # 判断开头是否以句子开头（大写字母或中文字符）
            starts_ok = bool(re.match(r"[\u4e00-\u9fffA-Z0-9#\"'「『]", c[0]))
            consist_scores.append(1.0 if (ends_natural and starts_ok) else 0.0)
        consistency = sum(consist_scores) / num if num > 0 else 0.0

        # --- 2. 边界质量 (boundary) ---
        # 方法：chunk 切分点是否在自然边界（段落分隔、标题行）
        # 对原始文本定位切分点，检查每个切分点是否落在自然边界
        boundary_hits = 0
        boundary_total = num - 1 if num > 1 else 1
        if num > 1:
            # 重建切分位置
            pos = 0
            for i, c in enumerate(chunks[:-1]):
                pos += len(c)
                # 检查切分点附近是否在自然边界
                around = original_text[max(0, pos - 5): min(len(original_text), pos + 5)]
                if self._PARA_BOUNDARY.search(around) or re.search(r"[。！？；\n]", around[:10]):
                    boundary_hits += 1
        else:
            boundary_hits = 1
        boundary = boundary_hits / boundary_total if boundary_total > 0 else 1.0

        # --- 3. 信息密度 (density) ---
        # 方法：有效字符（非连续空白、非重复噪音）占比
        total_chars = sum(len(c) for c in chunks)
        noise_chars = sum(len("".join(self._NOISE_PATTERN.findall(c))) for c in chunks)
        density = (total_chars - noise_chars) / total_chars if total_chars > 0 else 0.0

        # --- 4. 重叠利用率 (overlap_eff) ---
        # 方法：重叠区域的内容在相邻 chunk 中是否包含关键信息
        if overlap_size > 0 and num > 1:
            overlap_used = 0
            for i in range(num - 1):
                if len(chunks[i]) >= overlap_size and len(chunks[i + 1]) >= overlap_size:
                    tail = chunks[i][-overlap_size:]
                    head = chunks[i + 1][:overlap_size]
                    # 计算 Jaccard 相似度衡量 overlap 区域复用度
                    if tail and head:
                        intersection = len(set(tail) & set(head))
                        union = len(set(tail) | set(head))
                        if union > 0:
                            overlap_used += intersection / union
            overlap_eff = overlap_used / (num - 1) if num > 1 else 0.0
        else:
            overlap_eff = 1.0

        # --- 5. 长度稳定性 (length_stab) ---
        if avg > 0:
            variance = sum((s - avg) ** 2 for s in sizes) / num
            length_stab = math.sqrt(variance) / avg
        else:
            length_stab = 0.0

        # --- 6. 检索召回 (retrieval_recall) ---
        # 方法：核心术语是否完整出现在单个 chunk 内（不被切分打断）
        terms = self._TERM_PATTERN.findall(original_text)
        terms = list(set(t.lower() for t in terms))  # 去重
        # 过滤低频或通用术语（出现次数超过 20 的可能是通用词）
        term_freq = {}
        for t in terms:
            term_freq[t] = len(re.findall(re.escape(t), original_text, re.IGNORECASE))
        core_terms = [t for t in terms if 2 <= term_freq.get(t, 0) <= 20]

        if core_terms:
            recalled = 0
            for term in core_terms:
                # 检查是否有任一 chunk 包含完整术语
                found = any(term in c.lower() for c in chunks)
                if found:
                    recalled += 1
            retrieval_recall = recalled / len(core_terms)
        else:
            retrieval_recall = 1.0

        # --- 检查阈值 ---
        violations = []
        checks = {
            "consistency": consistency,
            "boundary": boundary,
            "density": density,
            "overlap_eff": overlap_eff,
            "length_stab": length_stab,
            "retrieval_recall": retrieval_recall,
        }
        for name, value in checks.items():
            threshold, msg = self.THRESHOLDS[name]
            is_stab = name == "length_stab"
            if (value > threshold if is_stab else value < threshold):
                violations.append(msg)

        return ChunkQualityReport(
            strategy_name=strategy_name,
            num_chunks=num, total_chars=total,
            consistency=round(consistency, 4),
            boundary=round(boundary, 4),
            density=round(density, 4),
            overlap_eff=round(overlap_eff, 4),
            length_stab=round(length_stab, 4),
            retrieval_recall=round(retrieval_recall, 4),
            overlap_ratio=round(overlap_ratio, 4),
            avg_chunk_size=round(avg, 1),
            min_chunk_size=min_s, max_chunk_size=max_s,
            split_time_ms=round(split_time_ms, 2),
            violations=violations,
        )

    def compare_strategies(
        self,
        text: str,
        splitters: dict,
        overlap_size: int = 0,
    ) -> dict:
        """对比多个切分策略"""
        results = {}
        for name, splitter in splitters.items():
            start = time.perf_counter()
            chunks = splitter(text)
            elapsed = (time.perf_counter() - start) * 1000
            if not chunks:
                results[name] = self.evaluate([], text, name, elapsed)
                continue
            results[name] = self.evaluate(
                [c.content if hasattr(c, "content") else c for c in chunks],
                text, name, elapsed, overlap_size,
            )
        return results
