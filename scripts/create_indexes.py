"""Neo4j index optimization script.

Creates B-Tree indexes for commonly queried properties and full-text indexes
for entity search. Run this script after initial data ingestion.

Usage:
    python scripts/create_indexes.py [--all|--entity|--chunk|--fulltext]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.core.neo4j_client import get_neo4j_client
from loguru import logger


INDEXES = {
    "entity_name_btree": {
        "query": "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "description": "B-Tree index on Entity.name for exact match lookups",
    },
    "entity_type_btree": {
        "query": "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "description": "B-Tree index on Entity.type for type filtering",
    },
    "entity_name_type_composite": {
        "query": "CREATE INDEX entity_name_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.name, e.type)",
        "description": "Composite index on Entity(name, type) for combined lookups",
    },
    "chunk_id_btree": {
        "query": "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        "description": "B-Tree index on Chunk.id for fast chunk lookups",
    },
    "chunk_document_id_btree": {
        "query": "CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        "description": "B-Tree index on Chunk.document_id for document-chunk joins",
    },
    "chunk_index_btree": {
        "query": "CREATE INDEX chunk_index_idx IF NOT EXISTS FOR (c:Chunk) ON (c.index)",
        "description": "B-Tree index on Chunk.index for ordered retrieval",
    },
    "document_id_btree": {
        "query": "CREATE INDEX document_id_idx IF NOT EXISTS FOR (d:Document) ON (d.id)",
        "description": "B-Tree index on Document.id for fast document lookups",
    },
    "document_content_hash_btree": {
        "query": "CREATE INDEX document_hash_idx IF NOT EXISTS FOR (d:Document) ON (d.content_hash)",
        "description": "B-Tree index on Document.content_hash for duplicate detection",
    },
    "entity_contains_entity_idx": {
        "query": "CREATE INDEX contains_entity_idx IF NOT EXISTS FOR ()-[r:CONTAINS_ENTITY]->() ON (r)",
        "description": "Index on CONTAINS_ENTITY relationship for entity-chunk joins",
    },
    "entity_has_chunk_idx": {
        "query": "CREATE INDEX has_chunk_idx IF NOT EXISTS FOR ()-[r:HAS_CHUNK]->() ON (r)",
        "description": "Index on HAS_CHUNK relationship for document-chunk joins",
    },
}


FULLTEXT_INDEXES = {
    "entity_fulltext": {
        "query": "CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.properties]",
        "description": "Full-text index on Entity.name and properties for fuzzy search",
    },
    "chunk_fulltext": {
        "query": "CREATE FULLTEXT INDEX chunk_fulltext_idx IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content]",
        "description": "Full-text index on Chunk.content for text search",
    },
}


def create_indexes(client, index_keys=None, include_fulltext=False):
    """Create specified indexes in Neo4j"""
    if index_keys is None:
        index_keys = list(INDEXES.keys())

    indexes_to_create = {k: INDEXES[k] for k in index_keys if k in INDEXES}

    if include_fulltext:
        indexes_to_create.update(FULLTEXT_INDEXES)

    logger.info(f"Creating {len(indexes_to_create)} indexes...")

    for name, index_info in indexes_to_create.items():
        try:
            logger.info(f"Creating index '{name}': {index_info['description']}")
            client.execute_query(index_info["query"])
            logger.info(f"  Index '{name}' created or already exists")
        except Exception as e:
            logger.warning(f"  Index '{name}' creation failed: {e}")

    logger.info("Index creation complete")


def show_existing_indexes(client):
    """Show existing indexes in Neo4j"""
    query = "SHOW INDEXES"
    results = client.execute_query(query)

    logger.info(f"Existing indexes ({len(results)}):")
    for row in results:
        name = row.get("name", "unknown")
        index_type = row.get("type", "unknown")
        state = row.get("state", "unknown")
        labels_or_types = row.get("labelsOrTypes", [])
        properties = row.get("properties", [])
        logger.info(
            f"  {name} ({index_type}, {state}) "
            f"on {labels_or_types} {properties}"
        )

    return results


def analyze_slow_queries(client):
    """Analyze and suggest optimizations for common queries"""
    queries = [
        {
            "name": "Entity lookup by name",
            "query": "MATCH (e:Entity {name: '高血压'}) RETURN e",
            "expected_index": "entity_name_idx",
        },
        {
            "name": "Entity lookup by type",
            "query": "MATCH (e:Entity:Disease) RETURN e LIMIT 10",
            "expected_index": "entity_type_idx",
        },
        {
            "name": "Chunk lookup by document",
            "query": "MATCH (c:Chunk {document_id: 'doc123'}) RETURN c ORDER BY c.index",
            "expected_index": "chunk_doc_id_idx",
        },
        {
            "name": "Document duplicate check",
            "query": "MATCH (d:Document {content_hash: 'abc123'}) RETURN d LIMIT 1",
            "expected_index": "document_hash_idx",
        },
    ]

    logger.info("Analyzing common queries with EXPLAIN:")
    for q in queries:
        try:
            explain_query = f"EXPLAIN {q['query']}"
            results = client.execute_query(explain_query)
            logger.info(f"  Query '{q['name']}': OK (expected index: {q['expected_index']})")
        except Exception as e:
            logger.warning(f"  Query '{q['name']}': EXPLAIN failed - {e}")


def main():
    parser = argparse.ArgumentParser(description="Create Neo4j indexes for optimization")
    parser.add_argument("--all", action="store_true", help="Create all indexes including full-text")
    parser.add_argument("--entity", action="store_true", help="Create entity-related indexes only")
    parser.add_argument("--chunk", action="store_true", help="Create chunk-related indexes only")
    parser.add_argument("--fulltext", action="store_true", help="Include full-text indexes")
    parser.add_argument("--show", action="store_true", help="Show existing indexes")
    parser.add_argument("--analyze", action="store_true", help="Analyze slow queries")

    args = parser.parse_args()

    client = get_neo4j_client()

    if not client.verify_connectivity():
        logger.error("Cannot connect to Neo4j")
        sys.exit(1)

    if args.show:
        show_existing_indexes(client)
        return

    if args.analyze:
        analyze_slow_queries(client)
        return

    if args.entity:
        index_keys = [k for k in INDEXES if "entity" in k]
        create_indexes(client, index_keys, args.fulltext)
    elif args.chunk:
        index_keys = [k for k in INDEXES if "chunk" in k or "document" in k]
        create_indexes(client, index_keys, args.fulltext)
    elif args.all:
        create_indexes(client, include_fulltext=args.fulltext)
    else:
        create_indexes(client, include_fulltext=True)

    if args.show:
        show_existing_indexes(client)


if __name__ == "__main__":
    main()
