"""
Grabber configuration via pydantic-settings.

All env vars loaded from environment or .env file.
Singleton access via get_settings().
"""

from __future__ import annotations

import functools
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required ---
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://temporal:temporal@localhost:5432/ghostgraph"

    # --- LLM (Henry — local Qwen3.5-35B via arcllm-proxy) ---
    henry_url: str = "http://localhost:11435/v1/chat/completions"
    henry_model: str = "qwen35"

    # --- Networking / Stealth ---
    ipv6_subnet: str = ""
    spoof_ipv6: bool = False  # True = generate IPv6 addresses but don't bind (local testing)
    scraping_mode: str = "handoff"  # light | handoff | full

    # --- Fleet ---
    vultr_api_key: str = ""
    github_deploy_key: str = ""
    api_base_url: str = "http://localhost:8000"
    worker_redis_url: str = ""
    worker_postgres_url: str = ""

    # --- Postgres connection pool ---
    db_pool_min: int = 1
    db_pool_max: int = 3

    # --- Task defaults ---
    default_deadline_minutes: int = 30

    # --- Stealth tuning ---
    timeout_chain: list[int] = [5, 10, 15]
    max_403_rotations: int = 5
    cookie_cache_ttl: int = 1500
    residential_daily_limit_gb: float = 1.0
    residential_cost_per_gb: float = 12.0
    default_rpm: int = 45
    cloudflare_rpm: int = 30
    akamai_rpm: int = 20
    prefer_firefox: bool = True
    xvfb_display: str = ":99"

    # --- Circuit breaker ---
    circuit_breaker_window: int = 20
    circuit_breaker_threshold: float = 0.6

    # --- Backpressure ---
    stream_soft_limit: int = 500
    stream_hard_limit: int = 1000

    # --- API Security ---
    api_require_keys: bool = False
    bootstrap_api_key: str = ""

    # --- Worker ---
    heartbeat_ttl: int = 120
    work_item_stale_minutes: int = 5
    reaper_interval_seconds: int = 60

    # --- Schema ---
    schema_cache_ttl_days: int = 14
    min_structured_fields: int = 4
    schema_sample_count: int = 3

    # --- Embedding ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
