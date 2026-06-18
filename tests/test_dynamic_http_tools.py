import httpx
import pytest
import respx

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import HttpToolConfig, HttpToolParam
from app.tools.dynamic_http_tools import call_dynamic_http_tool


def _state() -> CustomerServiceState:
    return CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="Order status",
    )


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_calls_post_with_headers_and_maps_response():
    route = respx.post(
        "https://mock.local/order",
        headers={"Authorization": "Bearer seller-token"},
        json={"order_id": "A100"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={"ok": True, "payload": {"status": "shipped"}},
        )
    )
    state = _state()
    config = HttpToolConfig(
        name="get_order",
        description="Get order",
        url="https://mock.local/order",
        method="POST",
        headers={"Authorization": "Bearer seller-token"},
        response_mapping={
            "success_field": "ok",
            "success_value": True,
            "data_field": "payload",
        },
        timeout=3.5,
        request_params=[
            HttpToolParam(name="order_id", description="Order id", required=True)
        ],
    )

    result = await call_dynamic_http_tool(state, config, {"order_id": "A100"})

    assert result == {"status": "shipped"}
    assert route.call_count == 1
    assert len(state.http_tool_results) == 1
    assert state.tool_call_history == [
        {
            "tool_name": "get_order",
            "arguments": {"order_id": "A100"},
            "result": {"status": "shipped"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "DELETE"])
@respx.mock
async def test_dynamic_http_tool_sends_read_method_params_in_query(method):
    route = respx.route(method=method, url="https://mock.local/resource").mock(
        return_value=httpx.Response(200, json={"code": 200, "data": {"found": True}})
    )
    config = HttpToolConfig(
        name=f"{method.lower()}_resource",
        description="Read resource",
        url="https://mock.local/resource",
        method=method,
    )

    result = await call_dynamic_http_tool(_state(), config, {"ids": ["A1", "A2"]})

    assert result == {"found": True}
    assert route.calls.last.request.url.params.get_list("ids") == ["A1", "A2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["PUT", "PATCH"])
@respx.mock
async def test_dynamic_http_tool_sends_write_method_params_as_json(method):
    route = respx.route(method=method, url="https://mock.local/resource").mock(
        return_value=httpx.Response(200, json={"code": 200, "data": {"updated": True}})
    )
    config = HttpToolConfig(
        name=f"{method.lower()}_resource",
        description="Update resource",
        url="https://mock.local/resource",
        method=method,
    )

    result = await call_dynamic_http_tool(
        _state(),
        config,
        {"changes": {"status": "closed"}, "tags": ["priority", "vip"]},
    )

    assert result == {"updated": True}
    assert route.calls.last.request.content == (
        b'{"changes":{"status":"closed"},"tags":["priority","vip"]}'
    )


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_cache_key_is_stable_for_nested_params():
    route = respx.post("https://mock.local/search").mock(
        return_value=httpx.Response(200, json={"code": 200, "data": {"count": 2}})
    )
    state = _state()
    config = HttpToolConfig(
        name="search_orders",
        description="Search orders",
        url="https://mock.local/search",
        method="POST",
    )

    first = await call_dynamic_http_tool(
        state,
        config,
        {
            "filters": {"status": ["paid", "shipped"], "region": "US"},
            "page": 1,
        },
    )
    second = await call_dynamic_http_tool(
        state,
        config,
        {
            "page": 1,
            "filters": {"region": "US", "status": ["paid", "shipped"]},
        },
    )

    assert first == second == {"count": 2}
    assert route.call_count == 1
    assert len(state.tool_call_history) == 2


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_records_missing_required_params():
    state = _state()
    config = HttpToolConfig(
        name="get_order",
        description="Get order",
        url="https://mock.local/order",
        method="GET",
        request_params=[
            HttpToolParam(name="order_id", description="Order id", required=True),
            HttpToolParam(name="include_items", description="Include items"),
        ],
    )

    result = await call_dynamic_http_tool(state, config, {"include_items": False})

    assert result == {
        "error": "missing_required_params",
        "missing": ["order_id"],
    }
    assert state.tool_call_history == [
        {
            "tool_name": "get_order",
            "arguments": {"include_items": False},
            "result": result,
        }
    ]
    assert not respx.calls


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_retries_transient_failure_with_exponential_backoff(
    monkeypatch,
):
    request = httpx.Request("GET", "https://mock.local/order")
    route = respx.get("https://mock.local/order").mock(
        side_effect=[
            httpx.ReadTimeout("slow upstream", request=request),
            httpx.Response(503, text="temporarily unavailable"),
            httpx.Response(200, json={"code": 200, "data": {"status": "shipped"}}),
        ]
    )
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("app.tools.dynamic_http_tools.asyncio.sleep", fake_sleep)
    state = _state()
    config = HttpToolConfig(
        name="get_order",
        description="Get order",
        url="https://mock.local/order",
        method="GET",
        timeout=0.1,
        max_retries=2,
        retry_backoff=0.25,
    )

    result = await call_dynamic_http_tool(state, config, {"order_id": "A100"})

    assert result == {"status": "shipped"}
    assert route.call_count == 3
    assert sleep_delays == [0.25, 0.5]
    assert len(state.tool_call_history) == 1


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_records_structured_error_after_retries_exhausted():
    route = respx.delete("https://mock.local/order").mock(
        return_value=httpx.Response(500, text="upstream failed")
    )
    state = _state()
    config = HttpToolConfig(
        name="delete_order",
        description="Delete order",
        url="https://mock.local/order",
        method="DELETE",
        max_retries=1,
        retry_backoff=0,
    )

    result = await call_dynamic_http_tool(state, config, {"order_id": "A100"})

    assert result["error"] == "http_tool_failed"
    assert "500 Internal Server Error" in result["message"]
    assert route.call_count == 2
    assert state.tool_call_history[-1]["result"] == result
