from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.compatibility import convert_emailv4_payload
from app.api.routes import _validate_webhook_url
from app.harness.business_agent_executor import (
    AgentRunResult,
    BusinessAgentExecutor,
)
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import CustomerServiceRequest, HttpToolConfig
from app.tools.dynamic_http_tools import call_dynamic_http_tool


def test_emailv4_requires_explicit_tenant():
    with pytest.raises(ValueError, match="corp"):
        convert_emailv4_payload({"content": "Help"})


@pytest.mark.parametrize("max_turns", [0, -1, 21, 1_000_000])
def test_customer_service_request_bounds_max_turns(max_turns):
    with pytest.raises(ValidationError):
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Help",
            max_turns=max_turns,
        )


@pytest.mark.asyncio
async def test_harness_sanitizes_internal_exception_details():
    async def failing_runner(*args, **kwargs):
        raise RuntimeError("secret-provider-token=abc123")

    response = await CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=failing_runner)
    ).run(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Help",
        )
    )

    serialized = response.model_dump_json()
    assert response.status == "error"
    assert response.error == "Customer service processing failed."
    assert "secret-provider-token" not in serialized
    error_event = next(
        event
        for event in response.state_snapshot["events"]
        if event["name"] == "request_error"
    )
    assert error_event["detail"] == {"error_type": "RuntimeError"}


@pytest.mark.asyncio
async def test_harness_rejects_model_outside_deployment_allowlist(monkeypatch):
    called = False

    async def runner(*args, **kwargs):
        nonlocal called
        called = True
        return AgentRunResult(final_output="Should not run")

    monkeypatch.setattr(
        "app.harness.customer_service_harness.get_settings",
        lambda: SimpleNamespace(
            allowed_models="gpt-4.1-mini,gpt-4.1",
        ),
    )
    response = await CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=runner)
    ).run(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Help",
            model="unbounded-expensive-model",
        )
    )

    assert called is False
    assert response.status == "error"
    assert response.error == "Customer service processing failed."


@pytest.mark.asyncio
async def test_dynamic_http_requires_https_in_production(monkeypatch):
    monkeypatch.setattr(
        "app.tools.dynamic_http_tools.get_settings",
        lambda: SimpleNamespace(
            environment="production",
            dynamic_http_allowed_hosts="api.example.com",
        ),
    )
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Help",
    )

    result = await call_dynamic_http_tool(
        state,
        HttpToolConfig(
            name="lookup",
            description="Lookup",
            url="http://api.example.com/orders",
        ),
        {},
    )

    assert result == {"error": "https_required"}


def test_webhook_requires_https_in_production(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.get_settings",
        lambda: SimpleNamespace(
            environment="production",
            webhook_allowed_hosts="hooks.example.com",
        ),
    )

    with pytest.raises(ValueError, match="HTTPS"):
        _validate_webhook_url("http://hooks.example.com/result")
