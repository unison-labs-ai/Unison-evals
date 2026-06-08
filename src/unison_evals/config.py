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
    # Secret-gated eval lifecycle. When set, the unison-agent adapter sends it as
    # the X-Unison-Eval header and provisions a fresh ephemeral tenant per question
    # (memory benches) — no JWT needed. Required to run reproducible, isolated
    # benchmarks against the deployed (prod) app.
    unison_eval_secret: str = ""

    # Anthropic (judge cost accounting)
    anthropic_api_key: str = ""

    # OpenAI (real judge: gpt-4o-2024-08-06 + Context-Bench gpt-5-mini)
    openai_api_key: str = ""

    # Google (dev/research judge: gemini-* — the default --dev judge)
    google_api_key: str = ""

    # Models
    judge_model: str = "claude-opus-4-5-20250101"
    # Unison SUT model override. EMPTY (default) = submit the task with NO model,
    # so the SERVER runs its production auto model path exactly like a live user
    # turn — the eval must not choose the model. Set a value ONLY for an explicit
    # ablation (e.g. "claude-sonnet-4-5"). Requires a server-side fix defaulting
    # eval-turn to auto model selection deployed.
    unison_agent_model: str = ""
    # Dev/research judge (cheap, on Gemini credits). Used by --dev runs in place
    # of the per-benchmark canonical judge. gemini-3.1-flash-lite: Google's
    # current-gen (June 2026) cost-efficient GA model, ~$0.0005/judge-call.
    # Bump to gemini-3.5-flash for higher grading fidelity if a calibration vs
    # the real gpt-4o judge shows drift (the judge cost is noise vs agent cost).
    dev_judge_model: str = "gemini-3.1-flash-lite"

    # Timeouts (seconds)
    # 600s ceiling so MemoryAgentBench's long-context tiers (197K-1M tokens,
    # which exceed 120s) just work. It's a ceiling, not a target — fast benches
    # still return in seconds.
    adapter_timeout: int = 600
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
