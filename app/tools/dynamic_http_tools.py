import asyncio
import hashlib
import ipaddress
import json
import socket
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import httpcore
import httpx

from app.config import get_settings
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import HttpToolConfig, HttpToolParam


_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "email",
    "password",
    "secret",
    "token",
}
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_IDEMPOTENT_METHODS = {"GET", "PUT", "DELETE"}
_LOCAL_DEVELOPMENT_HOSTS = {"mock.local", "localhost", "127.0.0.1"}


class _PinnedDNSBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self,
        addresses_by_host: dict[str, list[str]],
        backend: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._addresses_by_host = {
            host.casefold(): list(addresses)
            for host, addresses in addresses_by_host.items()
        }
        self._backend = backend or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = self._addresses_by_host.get(host.casefold())
        if not addresses:
            raise httpcore.ConnectError(
                f"No prevalidated address for host {host}"
            )

        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:
                last_error = exc
        raise httpcore.ConnectError(
            f"Could not connect to a prevalidated address for {host}"
        ) from last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._backend.connect_unix_socket(
            path,
            timeout=timeout,
            socket_options=socket_options,
        )

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PinnedDNSAsyncTransport(httpx.AsyncHTTPTransport):
    def __init__(self, host: str, addresses: list[str]) -> None:
        super().__init__(trust_env=False)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpcore.default_ssl_context(),
            network_backend=_PinnedDNSBackend({host: addresses}),
        )


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.casefold() in _SENSITIVE_KEYS
            else _safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + "...[TRUNCATED]"
    return value


def _record_call(
    state: CustomerServiceState,
    tool_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
    *,
    attempts: int,
    cache_hit: bool = False,
    status_code: int | None = None,
) -> None:
    state.tool_call_history.append(
        {
            "tool_name": tool_name,
            "arguments": _safe_value(params),
            "result": _safe_value(result),
            "metadata": {
                "attempts": attempts,
                "cache_hit": cache_hit,
                "status_code": status_code,
            },
        }
    )


def _cache_key(config: HttpToolConfig, params: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "name": config.name,
            "method": config.method,
            "url": config.url,
            "headers": config.headers,
            "response_mapping": config.response_mapping,
            "params": params,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{config.name}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _allowed_hosts() -> set[str]:
    settings = get_settings()
    configured = settings.dynamic_http_allowed_hosts
    if configured:
        return {
            host.strip().casefold()
            for host in configured.split(",")
            if host.strip()
        }
    if settings.environment in {"local", "test"}:
        return _LOCAL_DEVELOPMENT_HOSTS
    return set()


def _validate_target(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        return "invalid_url"
    if host not in _allowed_hosts():
        return "target_not_allowed"
    if (
        getattr(get_settings(), "environment", "local") == "production"
        and parsed.scheme != "https"
    ):
        return "https_required"
    return None


async def _resolve_host_addresses(host: str) -> list[str]:
    def resolve() -> list[str]:
        records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return sorted({str(record[4][0]) for record in records})

    return await asyncio.to_thread(resolve)


async def _validate_resolved_target(
    url: str,
) -> tuple[str | None, list[str]]:
    host = (urlparse(url).hostname or "").casefold()
    settings = get_settings()
    if (
        settings.environment in {"local", "test"}
        and host in _LOCAL_DEVELOPMENT_HOSTS
    ):
        return None, []

    try:
        addresses = await _resolve_host_addresses(host)
    except OSError:
        return "target_resolution_failed", []
    if not addresses:
        return "target_resolution_failed", []

    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                return "target_resolves_to_private_address", []
        except ValueError:
            return "target_resolution_failed", []
    return None, addresses


def _is_missing(params: dict[str, Any], name: str) -> bool:
    if name not in params:
        return True
    value = params[name]
    if value is None or value == "":
        return True
    return isinstance(value, (list, tuple, dict, set)) and not value


def _valid_param_type(param: HttpToolParam, value: Any) -> bool:
    if param.type == "string":
        return isinstance(value, str)
    if param.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if param.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if param.type == "boolean":
        return isinstance(value, bool)
    return False


def _validate_params(
    config: HttpToolConfig,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    declared = {param.name: param for param in config.request_params}
    if declared:
        unexpected = sorted(set(params) - set(declared))
        if unexpected:
            return {"error": "unexpected_params", "params": unexpected}

    missing = [
        param.name
        for param in config.request_params
        if param.required and _is_missing(params, param.name)
    ]
    if missing:
        return {"error": "missing_required_params", "missing": missing}

    invalid = [
        name
        for name, value in params.items()
        if name in declared
        and value is not None
        and not _valid_param_type(declared[name], value)
    ]
    if invalid:
        return {"error": "invalid_param_types", "params": invalid}
    return None


def _map_response(data: Any, mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {"value": _safe_value(data)}

    response_mapping = mapping or {}
    success_field = response_mapping.get("success_field", "code")
    success_value = response_mapping.get("success_value", 200)
    data_field = response_mapping.get("data_field", "data")
    if str(data.get(success_field)) != str(success_value):
        return {"error": "api_error", "raw": _safe_value(dict(data))}

    value = data.get(data_field, {})
    return value if isinstance(value, dict) else {"value": value}


def _can_retry(config: HttpToolConfig) -> bool:
    if config.method in _IDEMPOTENT_METHODS:
        return True
    return any(key.casefold() == "idempotency-key" for key in config.headers)


def _parse_success_response(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 204 or not response.content:
        return {}
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.casefold():
        return {"value": _safe_value(response.text)}
    try:
        return _map_response(response.json(), None)
    except ValueError:
        return {"value": _safe_value(response.text)}


async def call_dynamic_http_tool(
    state: CustomerServiceState,
    config: HttpToolConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    target_error = _validate_target(config.url)
    if target_error:
        result = {"error": target_error}
        _record_call(state, config.name, params, result, attempts=0)
        return result
    resolved_target_error, resolved_addresses = (
        await _validate_resolved_target(config.url)
    )
    if resolved_target_error:
        result = {"error": resolved_target_error}
        _record_call(state, config.name, params, result, attempts=0)
        return result

    validation_error = _validate_params(config, params)
    if validation_error:
        _record_call(state, config.name, params, validation_error, attempts=0)
        return validation_error

    request_params = {key: value for key, value in params.items() if value is not None}
    cache_key = _cache_key(config, request_params)
    if config.method == "GET" and cache_key in state.http_tool_results:
        result = state.http_tool_results[cache_key]
        _record_call(
            state,
            config.name,
            params,
            result,
            attempts=0,
            cache_hit=True,
        )
        return result

    retry_enabled = _can_retry(config)
    max_attempts = config.max_retries + 1 if retry_enabled else 1
    last_error = "unknown error"
    last_status: int | None = None

    host = (urlparse(config.url).hostname or "").casefold()
    transport = (
        _PinnedDNSAsyncTransport(host, resolved_addresses)
        if resolved_addresses
        else None
    )
    async with httpx.AsyncClient(
        timeout=config.timeout,
        transport=transport,
        trust_env=False,
    ) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.request(
                    config.method,
                    config.url,
                    params=request_params
                    if config.method in {"GET", "DELETE"}
                    else None,
                    json=request_params
                    if config.method in {"POST", "PUT", "PATCH"}
                    else None,
                    headers=config.headers,
                )
                last_status = response.status_code
                if response.is_success:
                    if response.status_code == 204 or not response.content:
                        result = {}
                    else:
                        try:
                            data = response.json()
                        except ValueError:
                            result = {"value": _safe_value(response.text)}
                        else:
                            result = _map_response(data, config.response_mapping)
                    if config.method == "GET" and "error" not in result:
                        state.http_tool_results[cache_key] = result
                    _record_call(
                        state,
                        config.name,
                        params,
                        result,
                        attempts=attempt,
                        status_code=response.status_code,
                    )
                    return result

                last_error = f"HTTP {response.status_code}"
                if (
                    response.status_code not in _RETRYABLE_STATUS_CODES
                    or attempt >= max_attempts
                ):
                    break
            except httpx.RequestError as exc:
                last_error = str(exc)
                if attempt >= max_attempts:
                    break

            await asyncio.sleep(config.retry_backoff * (2 ** (attempt - 1)))

    result = {"error": "http_tool_failed", "message": _safe_value(last_error)}
    _record_call(
        state,
        config.name,
        params,
        result,
        attempts=max_attempts,
        status_code=last_status,
    )
    return result
