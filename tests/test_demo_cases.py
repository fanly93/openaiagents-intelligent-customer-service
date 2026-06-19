import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.api.routes import get_harness, get_webhook_poster, post_webhook
from app.config import Settings
from app.main import app
from app.state.request_models import CustomerServiceRequest
from app.state.result_models import CustomerServiceResponse
from app.tools.dynamic_http_tools import _PinnedDNSAsyncTransport


class FakeHarness:
    def __init__(self, response: CustomerServiceResponse) -> None:
        self.response = response
        self.requests: list[CustomerServiceRequest] = []

    async def run(
        self,
        request: CustomerServiceRequest,
    ) -> CustomerServiceResponse:
        self.requests.append(request)
        return self.response.model_copy(update={"request_id": request.request_id})


class FailingHarness:
    async def run(
        self,
        request: CustomerServiceRequest,
    ) -> CustomerServiceResponse:
        raise RuntimeError("secret upstream credential")


WebhookPoster = Callable[[str, CustomerServiceResponse], Awaitable[None]]


@pytest.fixture(autouse=True)
def allow_test_webhook_host(monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {"webhook_allowed_hosts": "callback.example"},
        )(),
        raising=False,
    )


def _override_dependencies(
    harness: Any,
    webhook_poster: WebhookPoster | None = None,
) -> None:
    app.dependency_overrides[get_harness] = lambda: harness
    if webhook_poster is not None:
        app.dependency_overrides[get_webhook_poster] = lambda: webhook_poster


def _clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()


async def _noop_webhook(
    url: str,
    payload: CustomerServiceResponse,
) -> None:
    return None


def _success_response(body: str) -> CustomerServiceResponse:
    return CustomerServiceResponse(
        status="success",
        reply={"subject": "Re: Support", "body": body},
    )


def test_webhook_allowed_hosts_setting_reads_environment(monkeypatch):
    monkeypatch.setenv("WEBHOOK_ALLOWED_HOSTS", "hooks.example.com,other.example")

    settings = Settings()

    assert settings.webhook_allowed_hosts == "hooks.example.com,other.example"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_demo_cases_endpoint_lists_supported_cases():
    with TestClient(app) as client:
        response = client.get("/demo/cases")

    assert response.status_code == 200
    assert "english_order_lookup" in response.json()["cases"]
    assert "dynamic_http_tool" in response.json()["cases"]


def test_customer_service_sync_endpoint_awaits_harness():
    harness = FakeHarness(_success_response("Hello from the harness"))
    _override_dependencies(harness)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "request_id": "req-sync",
                    "tenant_id": "tenant_a",
                    "content": "Hello",
                    "async_mode": False,
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 200
    assert response.json()["request_id"] == "req-sync"
    assert response.json()["reply"] == {
        "subject": "Re: Support",
        "body": "Hello from the harness",
    }
    assert [request.content for request in harness.requests] == ["Hello"]


def test_customer_service_async_endpoint_requires_webhook_url():
    harness = FakeHarness(_success_response("Unused"))
    _override_dependencies(harness, _noop_webhook)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "tenant_id": "tenant_a",
                    "content": "Handle this later",
                    "async_mode": True,
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 422
    assert harness.requests == []


@pytest.mark.parametrize(
    "webhook_url",
    [
        "ftp://callback.example/result",
        "https://blocked.example/result",
    ],
)
def test_customer_service_async_endpoint_rejects_unsafe_webhook_url(webhook_url):
    harness = FakeHarness(_success_response("Unused"))
    _override_dependencies(harness, _noop_webhook)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "tenant_id": "tenant_a",
                    "content": "Handle this later",
                    "async_mode": True,
                    "webhook_url": webhook_url,
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 422
    assert harness.requests == []


def test_customer_service_async_endpoint_returns_http_202():
    harness = FakeHarness(_success_response("Completed"))
    _override_dependencies(harness, _noop_webhook)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "tenant_id": "tenant_a",
                    "content": "Handle this later",
                    "async_mode": True,
                    "webhook_url": "https://callback.example/result",
                },
            )
    finally:
        _clear_dependency_overrides()

    body = response.json()
    assert response.status_code == 202
    assert body["status"] == "accepted"
    assert body["request_id"]
    assert body["accepted_at"]
    assert len(harness.requests) == 1
    assert harness.requests[0].request_id == body["request_id"]


def test_async_endpoint_posts_complete_response_to_webhook():
    harness = FakeHarness(
        CustomerServiceResponse(
            status="success",
            reply={"subject": "Re: Support", "body": "Completed"},
            token_usage={"total_tokens": 12},
        )
    )
    callbacks: list[tuple[str, CustomerServiceResponse]] = []

    async def capture_webhook(
        url: str,
        payload: CustomerServiceResponse,
    ) -> None:
        callbacks.append((url, payload))

    _override_dependencies(harness, capture_webhook)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "request_id": "req-webhook",
                    "tenant_id": "tenant_a",
                    "content": "Notify me",
                    "async_mode": True,
                    "webhook_url": "https://callback.example/result",
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 202
    assert len(callbacks) == 1
    callback_url, callback_payload = callbacks[0]
    assert callback_url == "https://callback.example/result"
    assert callback_payload.model_dump() == {
        "request_id": "req-webhook",
        "status": "success",
        "reply": {"subject": "Re: Support", "body": "Completed"},
        "need_handoff_to_human": False,
        "handoff_type": "no_handoff",
        "handoff_reason": None,
        "agent_chain": ["customer_service_agent"],
        "token_usage": {"total_tokens": 12},
        "processing_time": 0.0,
        "error": None,
        "state_snapshot": {},
    }


def test_background_harness_failure_posts_sanitized_error_response():
    callbacks: list[CustomerServiceResponse] = []

    async def capture_webhook(
        url: str,
        payload: CustomerServiceResponse,
    ) -> None:
        callbacks.append(payload)

    _override_dependencies(FailingHarness(), capture_webhook)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "request_id": "req-failure",
                    "tenant_id": "tenant_a",
                    "content": "Notify me",
                    "async_mode": True,
                    "webhook_url": "https://callback.example/result",
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 202
    assert len(callbacks) == 1
    callback_payload = callbacks[0]
    assert callback_payload.status == "error"
    assert callback_payload.request_id == "req-failure"
    assert callback_payload.error
    assert "secret upstream credential" not in callback_payload.error


def test_webhook_failure_is_contained_by_background_job():
    harness = FakeHarness(_success_response("Completed despite callback failure"))
    callback_attempts: list[str] = []

    async def failing_webhook(
        url: str,
        payload: CustomerServiceResponse,
    ) -> None:
        callback_attempts.append(url)
        raise RuntimeError("callback unavailable")

    _override_dependencies(harness, failing_webhook)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.post(
                "/customer-service/respond",
                json={
                    "tenant_id": "tenant_a",
                    "content": "Notify me",
                    "async_mode": True,
                    "webhook_url": "https://callback.example/result",
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert callback_attempts == ["https://callback.example/result"]
    assert len(harness.requests) == 1


@pytest.mark.asyncio
async def test_post_webhook_rejects_private_dns_resolution(monkeypatch):
    async def resolve_private(host: str) -> list[str]:
        return ["10.0.0.8"]

    class NetworkMustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("network client must not be created")

    monkeypatch.setattr(routes, "_resolve_host_addresses", resolve_private)
    monkeypatch.setattr(routes.httpx, "AsyncClient", NetworkMustNotRun)

    with pytest.raises(ValueError, match="non-global"):
        await post_webhook(
            "https://callback.example/result",
            _success_response("Blocked"),
        )


@pytest.mark.asyncio
async def test_post_webhook_pins_prevalidated_ip_and_disables_environment(
    monkeypatch,
):
    async def resolve_global(host: str) -> list[str]:
        return ["93.184.216.34"]

    client_options: dict[str, Any] = {}
    posted: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            client_options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url: str, json: dict[str, Any]):
            posted.update({"url": url, "json": json})
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(routes, "_resolve_host_addresses", resolve_global)
    monkeypatch.setattr(routes.httpx, "AsyncClient", FakeAsyncClient)
    payload = _success_response("Delivered")

    await post_webhook("https://callback.example/result", payload)

    assert client_options["trust_env"] is False
    assert isinstance(client_options["transport"], _PinnedDNSAsyncTransport)
    assert posted == {
        "url": "https://callback.example/result",
        "json": payload.model_dump(mode="json"),
    }


def test_emailv4_endpoint_uses_compatibility_mapping():
    harness = FakeHarness(_success_response("Legacy request handled"))
    _override_dependencies(harness)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/emailv4",
                json={
                    "qid": "legacy-qid",
                    "corp": "legacy-tenant",
                    "channel": "chat",
                    "content": "Legacy payload",
                },
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 200
    assert response.json()["reply"] == {
        "subject": "Re: Support",
        "body": "Legacy request handled",
    }
    assert len(harness.requests) == 1
    request = harness.requests[0]
    assert request.request_id == "legacy-qid"
    assert request.tenant_id == "legacy-tenant"
    assert request.channel == "chat"
    assert request.content == "Legacy payload"


def test_emailv4_endpoint_returns_422_for_malformed_payload():
    harness = FakeHarness(_success_response("Unused"))
    _override_dependencies(harness)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/emailv4",
                json={"qid": "missing-content"},
            )
    finally:
        _clear_dependency_overrides()

    assert response.status_code == 422
    assert harness.requests == []


@pytest.mark.asyncio
async def test_asgi_response_is_sent_before_background_harness_completes():
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class BlockingHarness:
        async def run(
            self,
            request: CustomerServiceRequest,
        ) -> CustomerServiceResponse:
            started.set()
            await release.wait()
            finished.set()
            return _success_response("Completed")

    _override_dependencies(BlockingHarness(), _noop_webhook)
    request_body = json.dumps(
        {
            "tenant_id": "tenant_a",
            "content": "Handle this in the background",
            "async_mode": True,
            "webhook_url": "https://callback.example/result",
        }
    ).encode()
    receive_messages = [
        {
            "type": "http.request",
            "body": request_body,
            "more_body": False,
        }
    ]
    send_events: list[dict[str, Any]] = []
    response_sent = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if receive_messages:
            return receive_messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        send_events.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body",
            False,
        ):
            response_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/customer-service/respond",
        "raw_path": b"/customer-service/respond",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(response_sent.wait(), timeout=1)
        await asyncio.wait_for(started.wait(), timeout=1)

        assert send_events[0]["type"] == "http.response.start"
        assert send_events[0]["status"] == 202
        assert any(event["type"] == "http.response.body" for event in send_events)
        assert not finished.is_set()
        assert not task.done()
    finally:
        release.set()
        await task
        _clear_dependency_overrides()
