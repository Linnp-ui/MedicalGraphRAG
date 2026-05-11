import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.ingestion.kg_builder import KnowledgeGraphBuilder
from src.ingestion.document_loader import load_document

data_path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "input", "medical_sample.txt")
doc = load_document(data_path)
builder = KnowledgeGraphBuilder()
result = builder.ingest_document(doc, extract_entities=True, create_embeddings=True)
print(f"Entities: {result['entities_extracted']}")
print(f"Relationships: {result['relationships_created']}")
