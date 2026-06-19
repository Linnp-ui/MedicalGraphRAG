"""Tests for src.ingestion.pipeline"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import fields

from src.ingestion.pipeline import (
    ProcessingResult,
    BatchProcessingStats,
    DocumentProcessingPipeline,
    create_pipeline,
)


# ---------------------------------------------------------------------------
# TestProcessingResult
# ---------------------------------------------------------------------------
class TestProcessingResult:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(ProcessingResult)}
        assert field_names == {
            "success", "document", "chunks", "error",
            "file_path", "processing_time",
        }

    def test_defaults(self):
        r = ProcessingResult(success=True)
        assert r.document is None
        assert r.chunks == []
        assert r.error is None
        assert r.file_path is None
        assert r.processing_time == 0.0


# ---------------------------------------------------------------------------
# TestBatchProcessingStats
# ---------------------------------------------------------------------------
class TestBatchProcessingStats:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(BatchProcessingStats)}
        assert field_names == {
            "total_files", "successful", "failed", "skipped",
            "total_processing_time", "errors",
        }

    def test_defaults(self):
        s = BatchProcessingStats()
        assert s.total_files == 0
        assert s.successful == 0
        assert s.failed == 0
        assert s.skipped == 0
        assert s.total_processing_time == 0.0
        assert s.errors == []


# ---------------------------------------------------------------------------
# TestDocumentProcessingPipeline
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_deps():
    with patch("src.ingestion.pipeline.DocumentLoader") as mock_loader_cls, \
         patch("src.ingestion.pipeline.TextSplitter") as mock_splitter_cls, \
         patch("src.ingestion.pipeline.MedicalTextProcessor") as mock_proc_cls:

        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        mock_splitter = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_proc = MagicMock()
        mock_proc_cls.return_value = mock_proc

        yield {
            "loader": mock_loader,
            "splitter": mock_splitter,
            "processor": mock_proc,
        }


class TestDocumentProcessingPipeline:

    def test_process_document_success(self, mock_deps):
        mock_doc = MagicMock()
        mock_doc.content = "some medical text"
        mock_doc.id = "doc1"

        load_result = MagicMock()
        load_result.success = True
        load_result.document = mock_doc
        mock_deps["loader"].load_safe.return_value = load_result

        mock_deps["processor"].process_document.return_value = mock_doc
        mock_deps["splitter"].split_text.return_value = [MagicMock()]

        pipeline = DocumentProcessingPipeline(enable_medical_processing=True)
        result = pipeline.process_document("/path/to/file.txt")

        assert result.success is True
        assert result.document is mock_doc
        assert result.file_path == "/path/to/file.txt"
        assert result.processing_time >= 0

    def test_process_document_failure(self, mock_deps):
        load_result = MagicMock()
        load_result.success = False
        load_result.error = "File not found"
        mock_deps["loader"].load_safe.return_value = load_result

        pipeline = DocumentProcessingPipeline(enable_medical_processing=True)
        result = pipeline.process_document("/bad/path.txt")

        assert result.success is False
        assert result.error == "File not found"

    def test_process_batch(self, mock_deps):
        mock_doc = MagicMock()
        mock_doc.content = "text"
        mock_doc.id = "doc1"

        load_result = MagicMock()
        load_result.success = True
        load_result.document = mock_doc
        mock_deps["loader"].load_safe.return_value = load_result
        mock_deps["processor"].process_document.return_value = mock_doc
        mock_deps["splitter"].split_text.return_value = []

        pipeline = DocumentProcessingPipeline(enable_medical_processing=True)
        results, stats = pipeline.process_batch(
            ["/path1.txt", "/path2.txt"], show_progress=False
        )

        assert len(results) == 2
        assert stats.total_files == 2
        assert stats.successful == 2

    def test_shutdown(self, mock_deps):
        pipeline = DocumentProcessingPipeline()
        pipeline.shutdown()
        # No exception means success

    def test_create_pipeline_convenience(self):
        with patch("src.ingestion.pipeline.DocumentLoader"), \
             patch("src.ingestion.pipeline.TextSplitter"), \
             patch("src.ingestion.pipeline.MedicalTextProcessor"):
            pipeline = create_pipeline(
                max_workers=2, chunk_size=500,
                chunk_overlap=100, strategy="hybrid",
            )
            assert isinstance(pipeline, DocumentProcessingPipeline)
            assert pipeline.max_workers == 2
