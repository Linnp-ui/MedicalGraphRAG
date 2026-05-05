import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    extraction_model: str = Field(default="qwen-flash", alias="EXTRACTION_MODEL")

    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")

    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    split_strategy: str = Field(default="hybrid")
    keep_code_blocks: bool = Field(default=True)
    keep_headers: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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
