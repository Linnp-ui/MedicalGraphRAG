import pytest
from pathlib import Path


class TestDocumentLoader:
    """Test document loading functionality"""

    def test_load_text_file(self, tmp_path):
        """Test loading a text file"""
        from src.ingestion.document_loader import DocumentLoader

        # Create a test text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")

        loader = DocumentLoader()
        doc = loader.load(test_file)

        assert doc.title == "test"
        assert doc.content == "This is a test document."

    def test_load_batch(self, tmp_path):
        """Test loading multiple files"""
        from src.ingestion.document_loader import DocumentLoader

        # Create test files
        (tmp_path / "test1.txt").write_text("Document 1")
        (tmp_path / "test2.txt").write_text("Document 2")

        loader = DocumentLoader()
        docs = loader.load_batch(tmp_path)

        assert len(docs) == 2


class TestTextSplitter:
    """Test text splitting functionality"""

    def test_split_text(self):
        """Test basic text splitting"""
        from src.ingestion.text_splitter import TextSplitter

        text = "This is paragraph one.\n\nThis is paragraph two."
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text, "doc1")

        assert len(chunks) > 0

    def test_chunk_properties(self):
        """Test chunk properties"""
        from src.ingestion.text_splitter import TextSplitter

        text = "This is a test document with some content."
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split_text(text, "doc1")

        chunk = chunks[0]
        assert chunk.id.startswith("doc1_chunk_")
        assert chunk.document_id == "doc1"
        assert chunk.index >= 0
