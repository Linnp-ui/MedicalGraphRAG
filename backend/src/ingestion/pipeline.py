import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator
from loguru import logger

from .document_loader import Document, DocumentLoader, LoadResult
from .text_splitter import TextChunk, TextSplitter, SplitStrategy
from .medical_processor import MedicalTextProcessor


@dataclass
class ProcessingResult:
    success: bool
    document: Optional[Document] = None
    chunks: List[TextChunk] = field(default_factory=list)
    error: Optional[str] = None
    file_path: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class BatchProcessingStats:
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_processing_time: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)


class DocumentProcessingPipeline:
    """文档处理管道，支持异步批量处理"""

    def __init__(
        self,
        max_workers: int = 4,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        split_strategy: str = "hybrid",
        enable_medical_processing: bool = True,
    ):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=SplitStrategy(split_strategy),
        )
        self.medical_processor = MedicalTextProcessor() if enable_medical_processing else None

    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("DocumentProcessingPipeline shutdown completed")

    def process_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """同步处理单个文档

        Args:
            file_path: 文件路径

        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        file_path = str(file_path)

        try:
            result = self.loader.load_safe(file_path)

            if not result.success:
                return ProcessingResult(
                    success=False,
                    error=result.error,
                    file_path=file_path,
                    processing_time=time.time() - start_time,
                )

            document = result.document

            if self.medical_processor:
                document = self.medical_processor.process_document(document)

            chunks = self.text_splitter.split_text(document.content, document.id)

            return ProcessingResult(
                success=True,
                document=document,
                chunks=chunks,
                file_path=file_path,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                file_path=file_path,
                processing_time=time.time() - start_time,
            )

    async def process_document_async(self, file_path: Union[str, Path]) -> ProcessingResult:
        """异步处理单个文档

        Args:
            file_path: 文件路径

        Returns:
            ProcessingResult: 处理结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_document, file_path)

    def process_batch(
        self, file_paths: List[Union[str, Path]], show_progress: bool = True
    ) -> tuple[List[ProcessingResult], BatchProcessingStats]:
        """同步批量处理文档

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            (处理结果列表, 统计信息)
        """
        results = []
        stats = BatchProcessingStats(total_files=len(file_paths))

        try:
            from tqdm import tqdm
            iterator = tqdm(file_paths, desc="Processing documents") if show_progress else file_paths
        except ImportError:
            iterator = file_paths

        for file_path in iterator:
            result = self.process_document(file_path)
            results.append(result)

            if result.success:
                stats.successful += 1
            elif "skipped" in (result.error or "").lower():
                stats.skipped += 1
            else:
                stats.failed += 1
                stats.errors.append({
                    "file": str(file_path),
                    "error": result.error or "Unknown error",
                })

            stats.total_processing_time += result.processing_time

        stats.total_processing_time = sum(r.processing_time for r in results)
        return results, stats

    async def process_batch_async(
        self, file_paths: List[Union[str, Path]], show_progress: bool = True
    ) -> tuple[List[ProcessingResult], BatchProcessingStats]:
        """异步批量处理文档

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            (处理结果列表, 统计信息)
        """
        results = []
        stats = BatchProcessingStats(total_files=len(file_paths))

        tasks = [self.process_document_async(fp) for fp in file_paths]

        if show_progress:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)
                self._update_stats(stats, result)
                logger.info(f"Progress: {i + 1}/{len(tasks)} completed")
        else:
            results = await asyncio.gather(*tasks)
            for result in results:
                self._update_stats(stats, result)

        return results, stats

    def _update_stats(self, stats: BatchProcessingStats, result: ProcessingResult):
        """更新统计信息"""
        if result.success:
            stats.successful += 1
        elif result.error and "skip" in result.error.lower():
            stats.skipped += 1
        else:
            stats.failed += 1
            if result.error:
                stats.errors.append({
                    "file": result.file_path or "unknown",
                    "error": result.error,
                })

    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> tuple[List[ProcessingResult], BatchProcessingStats]:
        """处理目录中的所有文档

        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            extensions: 只处理指定扩展名的文件，如 [".pdf", ".docx"]
            show_progress: 是否显示进度

        Returns:
            (处理结果列表, 统计信息)
        """
        directory = Path(directory)
        file_paths = []

        patterns = ["*"] if not extensions else [f"*{ext}" for ext in extensions]

        for pattern in patterns:
            if recursive:
                file_paths.extend(directory.rglob(pattern))
            else:
                file_paths.extend(directory.glob(pattern))

        file_paths = [fp for fp in file_paths if fp.is_file()]

        logger.info(f"Found {len(file_paths)} files to process in {directory}")

        return self.process_batch(file_paths, show_progress=show_progress)

    def process_streaming(
        self, file_path: Union[str, Path]
    ) -> Generator[TextChunk, None, Optional[Document]]:
        """流式处理文档，边加载边分割

        适用于超大文档处理，减少内存占用

        Args:
            file_path: 文件路径

        Yields:
            TextChunk: 分割后的文本块

        Returns:
            Document: 文档对象（在所有块处理完成后返回）
        """
        file_path = Path(file_path)
        document = self.loader.load(file_path)

        if self.medical_processor:
            document = self.medical_processor.process_document(document)

        for chunk in self.text_splitter.split_text_streaming(document.content, document.id):
            yield chunk

        return document


def create_pipeline(
    max_workers: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "hybrid",
) -> DocumentProcessingPipeline:
    """创建文档处理管道

    Args:
        max_workers: 最大工作线程数
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        strategy: 分割策略

    Returns:
        DocumentProcessingPipeline: 文档处理管道实例
    """
    return DocumentProcessingPipeline(
        max_workers=max_workers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_strategy=strategy,
    )
