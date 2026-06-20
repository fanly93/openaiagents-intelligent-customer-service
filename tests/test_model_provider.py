from types import SimpleNamespace

import pytest

from app.config import Settings
from app.providers.openai_compatible import (
    ProviderConfigurationError,
    resolve_provider_config,
)


def provider_settings(**updates: object) -> SimpleNamespace:
    values = {
        "model_provider": "openai",
        "openai_api_key": "openai-test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-4.1-mini",
        "deepseek_api_key": "deepseek-test-key",
        "deepseek_base_url": "https://api.deepseek.com/v1",
        "deepseek_model": "deepseek-chat",
        "dashscope_api_key": "dashscope-test-key",
        "dashscope_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "dashscope_model": "qwen-plus",
        "model_timeout_seconds": 60.0,
        "model_max_retries": 1,
        "allowed_models": "gpt-4.1-mini,deepseek-chat,qwen-plus",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_settings_read_all_openai_compatible_provider_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.model_provider == "deepseek"
    assert settings.deepseek_api_key == "secret"
    assert settings.deepseek_model == "deepseek-chat"
    assert settings.deepseek_base_url == "https://api.deepseek.com/v1"


def test_provider_models_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in ("OPENAI_MODEL", "DEEPSEEK_MODEL", "DASHSCOPE_MODEL"):
        monkeypatch.delenv(env_name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_model is None
    assert settings.deepseek_model is None
    assert settings.dashscope_model is None


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("openai", "gpt-4.1-mini"),
        ("deepseek", "deepseek-chat"),
        ("dashscope", "qwen-plus"),
    ],
)
def test_resolver_selects_current_provider_model(
    provider: str,
    expected_model: str,
) -> None:
    config = resolve_provider_config(provider_settings(model_provider=provider))

    assert config.provider == provider
    assert config.model == expected_model
    assert config.timeout_seconds == 60.0
    assert config.max_retries == 1


def test_resolver_normalizes_timeout_and_retry_types() -> None:
    settings = provider_settings(
        model_timeout_seconds="60",
        model_max_retries="1",
    )

    config = resolve_provider_config(settings)

    assert config.timeout_seconds == 60.0
    assert type(config.timeout_seconds) is float
    assert config.max_retries == 1
    assert type(config.max_retries) is int


def test_unsupported_provider_raises_sanitized_error() -> None:
    settings = provider_settings(
        model_provider="unsupported",
        openai_api_key="do-not-leak-this-key",
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "unsupported_provider"
    assert "do-not-leak-this-key" not in str(exc_info.value)


def test_missing_api_key_reports_only_current_provider_field() -> None:
    settings = provider_settings(
        model_provider="deepseek",
        deepseek_api_key=" ",
        openai_api_key=None,
        dashscope_api_key=None,
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "missing_provider_fields"
    assert exc_info.value.fields == ("DEEPSEEK_API_KEY",)


def test_missing_provider_attributes_are_reported_as_missing_fields() -> None:
    settings = SimpleNamespace(
        model_provider="deepseek",
        model_timeout_seconds=60.0,
        model_max_retries=1,
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "missing_provider_fields"
    assert exc_info.value.fields == (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    )


@pytest.mark.parametrize("provider", ["openai", "deepseek", "dashscope"])
def test_missing_model_reports_only_current_provider_field(provider: str) -> None:
    settings = provider_settings(
        model_provider=provider,
        **{f"{provider}_model": None},
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "missing_provider_fields"
    assert exc_info.value.fields == (f"{provider.upper()}_MODEL",)


def test_file_base_url_is_rejected() -> None:
    settings = provider_settings(
        model_provider="openai",
        openai_base_url="file:///tmp/provider",
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "invalid_base_url"


def test_malformed_base_url_is_reported_as_invalid_base_url() -> None:
    settings = provider_settings(
        model_provider="openai",
        openai_base_url="https://[::1/v1",
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(settings)

    assert exc_info.value.code == "invalid_base_url"
    assert exc_info.value.fields == ("OPENAI_BASE_URL",)
