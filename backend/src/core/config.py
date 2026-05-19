import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:17687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")

    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(default="", alias="DASHSCOPE_BASE_URL")
    dashscope_model: str = Field(default="qwen-plus", alias="DASHSCOPE_MODEL")
    dashscope_temperature: float = Field(default=0.0, alias="DASHSCOPE_TEMPERATURE")
    dashscope_max_tokens: int = Field(default=2000, alias="DASHSCOPE_MAX_TOKENS")
    extraction_model: str = Field(default="qwen-flash", alias="EXTRACTION_MODEL")

    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    
    llm_provider: str = Field(default="dashscope", alias="LLM_PROVIDER")
    vector_provider: str = Field(default="neo4j", alias="VECTOR_PROVIDER")
    lancedb_path: str = Field(default="./lancedb", alias="LANCEDB_PATH")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    cache_backend: str = Field(default="redis", alias="CACHE_BACKEND")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_prefix: str = Field(default="graphrag", alias="REDIS_PREFIX")

    domain: Literal["general", "medical"] = Field(default="general", alias="DOMAIN")

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=75)
    split_strategy: str = Field(default="auto")
    keep_code_blocks: bool = Field(default=True)
    keep_headers: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    load_dotenv()
    return Settings()


def load_yaml_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    settings = get_settings()
    config = _replace_env_vars(config, settings)
    return config


def _replace_env_vars(config: Any, settings: Settings) -> Any:
    if isinstance(config, dict):
        return {k: _replace_env_vars(v, settings) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item, settings) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var_name = config[2:-1]
        return getattr(settings, var_name.lower(), config)
    return config


def load_cypher_queries() -> dict[str, Any]:
    queries_path = Path(__file__).parent.parent.parent / "config" / "cypher" / "queries.yaml"
    with open(queries_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts() -> dict[str, Any]:
    prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
