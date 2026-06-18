import asyncio
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import httpx

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import HttpToolConfig


def _cache_key(tool_name: str, params: dict[str, Any]) -> str:
    canonical_params = json.dumps(
        params,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_params.encode("utf-8")).hexdigest()
    return f"{tool_name}:{digest}"


def _is_missing(params: dict[str, Any], name: str) -> bool:
    if name not in params:
        return True

    value = params[name]
    if value is None or value == "":
        return True
    return isinstance(value, (list, tuple, dict, set)) and not value


def _map_response(data: Any, mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {
            "error": "invalid_response",
            "message": "HTTP tool response must be a JSON object",
            "raw": data,
        }

    response_mapping = mapping or {}
    success_field = response_mapping.get("success_field", "code")
    success_value = response_mapping.get("success_value", 200)
    data_field = response_mapping.get("data_field", "data")

    if str(data.get(success_field)) != str(success_value):
        return {
            "error": "api_error",
            "raw": dict(data),
        }

    value = data.get(data_field, {})
    return value if isinstance(value, dict) else {"value": value}


async def call_dynamic_http_tool(
    state: CustomerServiceState,
    config: HttpToolConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    missing = [
        param.name
        for param in config.request_params
        if param.required and _is_missing(params, param.name)
    ]
    if missing:
        result = {
            "error": "missing_required_params",
            "missing": missing,
        }
        state.record_tool_call(config.name, params, result)
        return result

    request_params = {key: value for key, value in params.items() if value is not None}
    key = _cache_key(config.name, request_params)
    if key in state.http_tool_results:
        result = state.http_tool_results[key]
        state.record_tool_call(config.name, params, result)
        return result

    last_error: str | None = None
    max_retries = max(0, config.max_retries)
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        for attempt in range(max_retries + 1):
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
                response.raise_for_status()
                result = _map_response(response.json(), config.response_mapping)
                if "error" not in result:
                    state.http_tool_results[key] = result
                state.record_tool_call(config.name, params, result)
                return result
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    delay = max(0, config.retry_backoff) * (2**attempt)
                    await asyncio.sleep(delay)

    result = {
        "error": "http_tool_failed",
        "message": last_error or "unknown error",
    }
    state.record_tool_call(config.name, params, result)
    return result
