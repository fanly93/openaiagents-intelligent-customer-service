from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "OpenAI Agents Customer Service"
    environment: Literal["local", "test", "production"] = "local"
    default_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")

    mock_data_dir: str = "data"
    rag_mode: Literal["mock", "remote"] = "mock"
    remote_rag_url: str | None = None
    dynamic_http_allowed_hosts: str | None = Field(
        default=None,
        alias="DYNAMIC_HTTP_ALLOWED_HOSTS",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
