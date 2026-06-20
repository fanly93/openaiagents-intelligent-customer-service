from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


SUPPORTED_PROVIDERS = ("openai", "deepseek", "dashscope")


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int


class ProviderConfigurationError(ValueError):
    def __init__(
        self,
        code: str,
        provider: str,
        fields: tuple[str, ...] = (),
    ) -> None:
        self.code = code
        self.provider = provider
        self.fields = fields
        field_text = f"; fields={', '.join(fields)}" if fields else ""
        super().__init__(f"Provider configuration error: {code}; provider={provider}{field_text}")


def resolve_provider_config(settings: Any) -> ProviderConfig:
    provider = str(settings.model_provider).strip().casefold()
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderConfigurationError("unsupported_provider", provider)

    field_prefix = provider.upper()
    api_key = getattr(settings, f"{provider}_api_key", None)
    base_url = getattr(settings, f"{provider}_base_url", None)
    model = getattr(settings, f"{provider}_model", None)
    provider_values = {
        f"{field_prefix}_API_KEY": api_key,
        f"{field_prefix}_BASE_URL": base_url,
        f"{field_prefix}_MODEL": model,
    }
    missing_fields = tuple(
        field_name
        for field_name, value in provider_values.items()
        if value is None or not str(value).strip()
    )
    if missing_fields:
        raise ProviderConfigurationError(
            "missing_provider_fields",
            provider,
            missing_fields,
        )

    normalized_base_url = str(base_url).strip().rstrip("/")
    try:
        parsed_base_url = urlparse(normalized_base_url)
    except ValueError as exc:
        raise ProviderConfigurationError(
            "invalid_base_url",
            provider,
            (f"{field_prefix}_BASE_URL",),
        ) from exc
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
        raise ProviderConfigurationError(
            "invalid_base_url",
            provider,
            (f"{field_prefix}_BASE_URL",),
        )

    return ProviderConfig(
        provider=provider,
        api_key=str(api_key).strip(),
        base_url=normalized_base_url,
        model=str(model).strip(),
        timeout_seconds=float(settings.model_timeout_seconds),
        max_retries=int(settings.model_max_retries),
    )
