# pyright: reportMissingImports=false
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфігурація веб-сервісу з підтримкою змінних оточення."""

    index_path: Path = Path("index.bin")
    corpus_dir: Path = Path("data")
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


def get_settings() -> Settings:
    return Settings()
