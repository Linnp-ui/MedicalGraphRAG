"""Pure unit tests for backend/src/core/neo4j_client.py

All Neo4j interactions are mocked via unittest.mock.patch.
No real Neo4j connection, no LLM API calls.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings
from src.core.neo4j_client import Neo4jClient, _SCHEMA_CACHE_TTL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    defaults = dict(
        neo4j_uri="bolt://localhost:17687",
        neo4j_username="neo4j",
        neo4j_password="password",
    )
    defaults.update(overrides)
    return Settings.model_construct(**defaults)


def _make_client(**settings_overrides):
    """Create a Neo4jClient with a Settings instance (no get_settings call)."""
    settings = _make_settings(**settings_overrides)
    return Neo4jClient(settings=settings)


# ---------------------------------------------------------------------------
# TestNormalizeIdentifier
# ---------------------------------------------------------------------------
class TestNormalizeIdentifier:
    """Tests for _normalize_identifier validation against regex."""

    def test_valid_identifier(self):
        client = _make_client()
        assert client._normalize_identifier("Person", "node_label") == "Person"

    def test_invalid_identifier_special_chars(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Invalid node_label"):
            client._normalize_identifier("Per@son!", "node_label")

    def test_invalid_identifier_starts_with_number(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Invalid node_label"):
            client._normalize_identifier("1Person", "node_label")

    def test_valid_long_identifier(self):
        client = _make_client()
        # 128 chars: starts with letter, followed by 127 alphanumerics
        ident = "A" + "a" * 127
        assert client._normalize_identifier(ident, "node_label") == ident

    def test_invalid_too_long_identifier(self):
        client = _make_client()
        # 129 chars: exceeds the 128-char limit (1 + 127 pattern)
        ident = "A" + "a" * 128
        with pytest.raises(ValueError, match="Invalid node_label"):
            client._normalize_identifier(ident, "node_label")

    def test_injection_attempt_with_backtick(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Invalid node_label"):
            client._normalize_identifier("Person`; MATCH (n) DETACH DELETE n; //", "node_label")

    def test_injection_attempt_with_semicolon(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Invalid node_label"):
            client._normalize_identifier("Person; DROP DATABASE", "node_label")

    def test_valid_underscore_start(self):
        client = _make_client()
        assert client._normalize_identifier("_Private", "node_label") == "_Private"

    def test_valid_single_char(self):
        client = _make_client()
        assert client._normalize_identifier("A", "node_label") == "A"


# ---------------------------------------------------------------------------
# TestNeo4jClientInit
# ---------------------------------------------------------------------------
class TestNeo4jClientInit:
    """Tests for Neo4jClient.__init__."""

    def test_init_with_settings(self):
        settings = _make_settings()
        client = Neo4jClient(settings=settings)
        assert client.settings is settings
        assert client._driver is None
        assert client._async_driver is None

    @patch("src.core.neo4j_client.get_settings")
    def test_init_without_settings_uses_get_settings(self, mock_get_settings):
        mock_settings = _make_settings()
        mock_get_settings.return_value = mock_settings
        client = Neo4jClient()
        mock_get_settings.assert_called_once()
        assert client.settings is mock_settings


# ---------------------------------------------------------------------------
# TestSchemaCache
# ---------------------------------------------------------------------------
class TestSchemaCache:
    """Tests for schema caching logic."""

    def setup_method(self):
        # Reset the module-level schema cache before each test
        import src.core.neo4j_client as mod
        mod._schema_cache = None

    def test_invalidate_schema_cache(self):
        import src.core.neo4j_client as mod
        mod._schema_cache = ("cached_schema", time.time() + 300)
        client = _make_client()
        client.invalidate_schema_cache()
        assert mod._schema_cache is None

    def test_get_schema_uses_cache(self):
        import src.core.neo4j_client as mod
        # Pre-populate cache with a non-expired entry
        cached_schema = "Node labels:\n  - Person\n\nRelationship types:\n  - KNOWS"
        mod._schema_cache = (cached_schema, time.time() + _SCHEMA_CACHE_TTL)

        client = _make_client()
        # execute_query should NOT be called when cache is valid
        with patch.object(client, "execute_query") as mock_exec:
            result = client.get_schema(use_cache=True)
        mock_exec.assert_not_called()
        assert result == cached_schema


# ---------------------------------------------------------------------------
# TestExecuteQuery
# ---------------------------------------------------------------------------
class TestExecuteQuery:
    """Tests for execute_query with mocked Neo4j session."""

    def test_execute_query_success(self):
        client = _make_client()

        mock_record = MagicMock()
        mock_record.data.return_value = {"name": "Alice", "age": 30}

        mock_session_obj = MagicMock()
        mock_session_obj.run.return_value = [mock_record]
        mock_session_obj.__enter__ = MagicMock(return_value=mock_session_obj)
        mock_session_obj.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session_obj

        with patch.object(client, "_get_driver", return_value=mock_driver), \
             patch("src.core.metrics.get_metrics_middleware", side_effect=Exception("no metrics")):
            result = client.execute_query("MATCH (n) RETURN n LIMIT 1")

        assert len(result) == 1
        assert result[0] == {"name": "Alice", "age": 30}

    def test_execute_query_with_parameters(self):
        client = _make_client()

        mock_record = MagicMock()
        mock_record.data.return_value = {"name": "Bob"}

        mock_session_obj = MagicMock()
        mock_session_obj.run.return_value = [mock_record]
        mock_session_obj.__enter__ = MagicMock(return_value=mock_session_obj)
        mock_session_obj.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session_obj

        with patch.object(client, "_get_driver", return_value=mock_driver), \
             patch("src.core.metrics.get_metrics_middleware", side_effect=Exception("no metrics")):
            params = {"name": "Bob"}
            result = client.execute_query("MATCH (n {name: $name}) RETURN n", parameters=params)

        mock_session_obj.run.assert_called_once_with(
            "MATCH (n {name: $name}) RETURN n", params
        )
        assert result[0] == {"name": "Bob"}


# ---------------------------------------------------------------------------
# TestVerifyConnectivity
# ---------------------------------------------------------------------------
class TestVerifyConnectivity:
    """Tests for verify_connectivity with mocked session."""

    def test_verify_success(self):
        client = _make_client()

        mock_result = MagicMock()
        mock_result.consume.return_value = None

        mock_session_obj = MagicMock()
        mock_session_obj.run.return_value = mock_result
        mock_session_obj.__enter__ = MagicMock(return_value=mock_session_obj)
        mock_session_obj.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session_obj

        with patch.object(client, "_get_driver", return_value=mock_driver):
            assert client.verify_connectivity() is True

    def test_verify_failure(self):
        client = _make_client()

        mock_session_obj = MagicMock()
        mock_session_obj.run.side_effect = Exception("Connection refused")
        mock_session_obj.__enter__ = MagicMock(return_value=mock_session_obj)
        mock_session_obj.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session_obj

        with patch.object(client, "_get_driver", return_value=mock_driver):
            assert client.verify_connectivity() is False


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------
class TestClose:
    """Tests for close() clearing drivers."""

    def test_close_clears_drivers(self):
        client = _make_client()

        mock_driver = MagicMock()
        mock_async_driver = MagicMock()
        client._driver = mock_driver
        client._async_driver = mock_async_driver

        client.close()

        mock_driver.close.assert_called_once()
        mock_async_driver.close.assert_called_once()
        assert client._driver is None
        assert client._async_driver is None

    def test_close_when_no_drivers(self):
        client = _make_client()
        # Should not raise even when drivers are None
        client.close()
        assert client._driver is None
        assert client._async_driver is None
