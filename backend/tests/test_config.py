"""Pure unit tests for backend/src/core/config.py

All external dependencies (file I/O, env vars) are mocked.
No Neo4j, no LLM API calls.
"""

import os
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from src.core.config import (
    Settings,
    _replace_env_vars,
    get_settings,
    load_cypher_queries,
    load_prompts,
    load_yaml_config,
)


class _SettingsStub:
    """Simple stub that mimics Settings for _replace_env_vars tests.

    _replace_env_vars uses getattr(settings, var_name.lower(), default),
    so we just need an object that supports getattr with lowercase field names
    and returns the default for unknown attributes.
    """

    _defaults = dict(
        neo4j_uri="bolt://localhost:17687",
        neo4j_username="neo4j",
        neo4j_password="password",
        dashscope_api_key="",
        dashscope_model="qwen-plus",
        dashscope_temperature=0.0,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        domain="general",
        cache_enabled=True,
        cache_ttl=3600,
        community_algorithm="leiden",
        community_levels=3,
        llm_cache_enabled=True,
        llm_cache_ttl=604800,
    )

    def __init__(self, **overrides):
        self._values = {**self._defaults, **overrides}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# TestSettings
# ---------------------------------------------------------------------------
class TestSettings:
    """Tests for the Settings pydantic model default values and validation."""

    def test_default_values(self):
        # Use model_construct to bypass env-file loading and alias resolution
        # This gives us an instance with only the defaults we specify
        s = Settings.model_construct(
            neo4j_uri="bolt://localhost:17687",
            neo4j_username="neo4j",
            neo4j_password="password",
            dashscope_api_key="",
            dashscope_base_url="",
            dashscope_model="qwen-plus",
            dashscope_temperature=0.0,
            dashscope_max_tokens=2000,
            extraction_model="qwen-flash",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            embedding_api_key="",
            embedding_base_url="",
            llm_provider="dashscope",
            vector_provider="neo4j",
            lancedb_path="./lancedb",
            app_host="0.0.0.0",
            app_port=8000,
            debug=True,
            cache_enabled=True,
            cache_ttl=3600,
            cache_backend="redis",
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password="",
            redis_prefix="graphrag",
            domain="general",
            community_algorithm="leiden",
            community_levels=3,
            community_resolution=1.0,
            community_min_size=5,
            llm_cache_enabled=True,
            llm_cache_ttl=604800,
            llm_cache_max_size=1000,
            cors_origins=["*"],
            cors_allow_credentials=False,
            chunk_size=512,
            chunk_overlap=75,
            split_strategy="auto",
            keep_code_blocks=True,
            keep_headers=True,
        )
        assert s.neo4j_uri == "bolt://localhost:17687"
        assert s.neo4j_username == "neo4j"
        assert s.neo4j_password == "password"
        assert s.dashscope_api_key == ""
        assert s.dashscope_model == "qwen-plus"
        assert s.dashscope_temperature == 0.0
        assert s.embedding_model == "text-embedding-3-small"
        assert s.embedding_dimensions == 1536
        assert s.domain == "general"
        assert s.cache_enabled is True
        assert s.cache_ttl == 3600
        assert s.community_algorithm == "leiden"
        assert s.community_levels == 3
        assert s.llm_cache_enabled is True
        assert s.llm_cache_ttl == 604800

    def test_domain_validation(self):
        # Valid domains via model_construct
        s1 = Settings.model_construct(domain="general")
        assert s1.domain == "general"
        s2 = Settings.model_construct(domain="medical")
        assert s2.domain == "medical"

        # Invalid domain should raise when using normal constructor
        with pytest.raises(Exception):
            Settings(domain="invalid_domain")

    def test_custom_values(self):
        s = Settings.model_construct(
            neo4j_uri="bolt://remote:7687",
            neo4j_username="admin",
            neo4j_password="secret",
            dashscope_api_key="sk-123",
            dashscope_model="qwen-max",
            dashscope_temperature=0.7,
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            domain="medical",
            cache_enabled=False,
            cache_ttl=7200,
            community_algorithm="louvain",
            community_levels=5,
            llm_cache_enabled=False,
            llm_cache_ttl=86400,
        )
        assert s.neo4j_uri == "bolt://remote:7687"
        assert s.neo4j_username == "admin"
        assert s.neo4j_password == "secret"
        assert s.dashscope_api_key == "sk-123"
        assert s.dashscope_model == "qwen-max"
        assert s.dashscope_temperature == 0.7
        assert s.embedding_model == "text-embedding-3-large"
        assert s.embedding_dimensions == 3072
        assert s.domain == "medical"
        assert s.cache_enabled is False
        assert s.cache_ttl == 7200
        assert s.community_algorithm == "louvain"
        assert s.community_levels == 5
        assert s.llm_cache_enabled is False
        assert s.llm_cache_ttl == 86400


# ---------------------------------------------------------------------------
# TestReplaceEnvVars
# ---------------------------------------------------------------------------
class TestReplaceEnvVars:
    """Tests for _replace_env_vars recursive env-var replacement."""

    def test_replace_string_var(self):
        settings = _SettingsStub(neo4j_password="my_secret")
        result = _replace_env_vars("${neo4j_password}", settings)
        assert result == "my_secret"

    def test_replace_nested_dict(self):
        settings = _SettingsStub(neo4j_username="admin", neo4j_password="s3cret")
        config = {
            "db": {
                "user": "${neo4j_username}",
                "pass": "${neo4j_password}",
            }
        }
        result = _replace_env_vars(config, settings)
        assert result == {"db": {"user": "admin", "pass": "s3cret"}}

    def test_replace_list(self):
        settings = _SettingsStub(neo4j_username="admin")
        config = ["${neo4j_username}", "plain_text", {"key": "${neo4j_username}"}]
        result = _replace_env_vars(config, settings)
        assert result == ["admin", "plain_text", {"key": "admin"}]

    def test_no_replacement_for_plain_strings(self):
        settings = _SettingsStub()
        config = {"host": "localhost", "port": 8080, "debug": True}
        result = _replace_env_vars(config, settings)
        assert result == {"host": "localhost", "port": 8080, "debug": True}

    def test_unknown_var_kept_as_is(self):
        settings = _SettingsStub()
        result = _replace_env_vars("${nonexistent_var}", settings)
        # getattr returns the default (the original string), so it stays unchanged
        assert result == "${nonexistent_var}"


# ---------------------------------------------------------------------------
# TestLoadYamlConfig
# ---------------------------------------------------------------------------
class TestLoadYamlConfig:
    """Tests for load_yaml_config with mocked file I/O."""

    def test_loads_config_file(self):
        yaml_content = "key1: value1\nkey2: value2\n"
        mock_settings = _SettingsStub()
        with patch("builtins.open", mock_open(read_data=yaml_content)), \
             patch.object(yaml, "safe_load", return_value={"key1": "value1", "key2": "value2"}), \
             patch("src.core.config.get_settings", return_value=mock_settings):
            result = load_yaml_config(Path("/fake/settings.yaml"))
            assert result["key1"] == "value1"
            assert result["key2"] == "value2"

    def test_replaces_env_vars_in_config(self):
        raw_config = {
            "database": {
                "uri": "${neo4j_uri}",
                "username": "${neo4j_username}",
                "password": "${neo4j_password}",
            },
            "plain": "untouched",
        }
        mock_settings = _SettingsStub(
            neo4j_uri="bolt://prod:7687",
            neo4j_username="prod_user",
            neo4j_password="prod_pass",
        )
        with patch("builtins.open", mock_open(read_data="")), \
             patch.object(yaml, "safe_load", return_value=raw_config), \
             patch("src.core.config.get_settings", return_value=mock_settings):
            result = load_yaml_config(Path("/fake/settings.yaml"))
        assert result["database"]["uri"] == "bolt://prod:7687"
        assert result["database"]["username"] == "prod_user"
        assert result["database"]["password"] == "prod_pass"
        assert result["plain"] == "untouched"


# ---------------------------------------------------------------------------
# TestGetSettings
# ---------------------------------------------------------------------------
class TestGetSettings:
    """Tests for the cached get_settings function."""

    def setup_method(self):
        # Clear the lru_cache before each test
        get_settings.cache_clear()

    def test_returns_settings_instance(self):
        with patch("src.core.config.load_dotenv"):
            result = get_settings()
        assert isinstance(result, Settings)

    def test_cached_instance_returned(self):
        with patch("src.core.config.load_dotenv"):
            s1 = get_settings()
            s2 = get_settings()
        assert s1 is s2
