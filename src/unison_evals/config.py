"""Settings — read from environment or .env file via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Unison API
    unison_api_url: str = "http://localhost:3001"
    unison_jwt: str = ""
    # Secret-gated eval lifecycle (ADR-0008). When set, the unison-agent adapter
    # sends it as the X-Unison-Eval header and provisions a fresh ephemeral
    # tenant per question (memory benches) — no Supabase JWT needed. Required to
    # run reproducible, isolated benchmarks against the deployed (prod) app.
    unison_eval_secret: str = ""

    # Anthropic (judge + claude-code adapter cost accounting)
    anthropic_api_key: str = ""

    # OpenAI (pgvector_naive embeddings; optional GPT judge)
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536  # text-embedding-3-small native dim

    # Mem0 cloud
    mem0_api_key: str = ""

    # Letta cloud / self-hosted
    letta_api_key: str = ""
    letta_base_url: str = ""  # leave empty for Letta cloud
    letta_agent_model: str = "openai/gpt-4o-mini"
    letta_agent_embedding: str = "openai/text-embedding-3-small"

    # Zep cloud (or self-hosted via zep_base_url)
    zep_api_key: str = ""
    zep_base_url: str = ""  # blank = Zep cloud (api.getzep.com)
    zep_ingest_wait_seconds: float = 10.0  # wait for async graph build after ingest

    # pgvector_naive — local Postgres DSN with pgvector extension
    pgvector_dsn: str = "postgres://postgres:evals@localhost:5433/postgres"

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # OpenAI chat (separate from embeddings / pgvector_naive)
    openai_chat_model: str = "gpt-5"

    # Models
    judge_model: str = "claude-opus-4-5-20250101"
    default_agent_model: str = "claude-sonnet-4-5"
    # Dev/research judge (cheap, on Gemini credits). Used by --dev runs in place
    # of the per-benchmark canonical judge. gemini-2.5-flash: smart enough to
    # track the real judge, ~$0.0007/call (≈ half of gpt-4o, ~4× of flash-lite).
    dev_judge_model: str = "gemini-2.5-flash"

    # Timeouts (seconds)
    adapter_timeout: int = 120
    judge_timeout: int = 60

    # Concurrency
    max_concurrent_questions: int = 3

    # Server
    server_port: int = 8001

    # Paths
    cache_dir: Path = Path.home() / ".cache" / "unison-evals"
    results_dir: Path = Path("results")

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
