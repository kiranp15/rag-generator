"""Runtime configuration, loaded from environment variables / .env.

Kept as a single small module (rather than pydantic-settings) so the core
pipeline can be imported and unit-tested without pydantic installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv isn't a hard requirement for the core pipeline / tests.
    pass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    data_dir: str = os.environ.get("DATA_DIR", "data")
    chunk_size: int = _int_env("CHUNK_SIZE", 800)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 120)
    top_k: int = _int_env("TOP_K", 5)


settings = Settings()
