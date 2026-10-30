"""Runtime configuration loaded from environment variables (prefix ``SYSREV_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. See ``.env.example`` for documentation of each field."""

    model_config = SettingsConfigDict(
        env_prefix="SYSREV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./data/sysrev.db"
    data_dir: Path = Path("./data")

    ncbi_email: str | None = None
    ncbi_tool: str = "sysrev-meta-analysis"
    ncbi_api_key: SecretStr | None = None

    europepmc_requests_per_second: float = Field(default=5.0, gt=0)

    openalex_mailto: str | None = None
    openalex_api_key: SecretStr | None = None
    openalex_requests_per_second: float = Field(default=8.0, gt=0)

    unpaywall_email: str | None = None
    unpaywall_requests_per_second: float = Field(default=5.0, gt=0)

    crossref_mailto: str | None = None
    crossref_requests_per_second: float = Field(default=5.0, gt=0)

    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=5, ge=0, le=10)

    # LLM provider keys: read from the environment or .env (never logged or stored).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    dedup_title_threshold: float = Field(default=95.0, ge=50.0, le=100.0)
    cohort_overlap_flag_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    @property
    def ncbi_requests_per_second(self) -> float:
        """NCBI E-utilities policy: 3 req/s without an API key, 10 req/s with one."""
        return 10.0 if self.ncbi_api_key else 3.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
