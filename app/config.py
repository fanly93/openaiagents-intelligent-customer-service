from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "OpenAI Agents Customer Service"
    environment: Literal["local", "test", "production"] = "local"
    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")
    run_live_model_tests: bool = Field(default=False, alias="RUN_LIVE_MODEL_TESTS")
    default_model: str = "gpt-4.1-mini"
    allowed_models: str = Field(
        default="gpt-4.1-mini",
        alias="ALLOWED_MODELS",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str | None = Field(default=None, alias="DEEPSEEK_MODEL")
    dashscope_api_key: str | None = Field(default=None, alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_model: str | None = Field(default=None, alias="DASHSCOPE_MODEL")
    model_timeout_seconds: float = Field(
        default=60.0,
        ge=1,
        le=300,
        alias="MODEL_TIMEOUT_SECONDS",
    )
    model_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        alias="MODEL_MAX_RETRIES",
    )

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
    webhook_allowed_hosts: str | None = Field(
        default=None,
        alias="WEBHOOK_ALLOWED_HOSTS",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
