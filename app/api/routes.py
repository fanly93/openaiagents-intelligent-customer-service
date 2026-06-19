import logging
import ipaddress
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Response,
    status,
)

from app.api.compatibility import convert_emailv4_payload
from app.config import get_settings
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.request_models import CustomerServiceRequest
from app.state.result_models import AcceptedResponse, CustomerServiceResponse
from app.tools.dynamic_http_tools import (
    _PinnedDNSAsyncTransport,
    _resolve_host_addresses,
)

logger = logging.getLogger(__name__)
router = APIRouter()

WebhookPoster = Callable[[str, CustomerServiceResponse], Awaitable[None]]


def get_harness() -> CustomerServiceHarness:
    return CustomerServiceHarness()


def _webhook_allowed_hosts() -> set[str]:
    configured = get_settings().webhook_allowed_hosts
    if not configured:
        return set()
    return {
        host.strip().casefold()
        for host in configured.split(",")
        if host.strip()
    }


def _validate_webhook_url(webhook_url: str) -> str:
    parsed = urlparse(webhook_url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("Webhook URL must use http or https.")
    if (
        getattr(get_settings(), "environment", "local") == "production"
        and parsed.scheme != "https"
    ):
        raise ValueError("Webhook URL must use HTTPS in production.")
    if host not in _webhook_allowed_hosts():
        raise ValueError("Webhook host is not allowed.")
    return host


async def _resolve_webhook_addresses(host: str) -> list[str]:
    try:
        addresses = await _resolve_host_addresses(host)
    except OSError as exc:
        raise ValueError("Webhook host resolution failed.") from exc
    if not addresses:
        raise ValueError("Webhook host resolution failed.")

    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError("Webhook host resolution failed.") from exc
        if not parsed_address.is_global:
            raise ValueError(
                "Webhook host resolved to a non-global address."
            )
    return addresses


async def post_webhook(
    webhook_url: str,
    response: CustomerServiceResponse,
) -> None:
    host = _validate_webhook_url(webhook_url)
    addresses = await _resolve_webhook_addresses(host)
    transport = _PinnedDNSAsyncTransport(host, addresses)
    async with httpx.AsyncClient(
        transport=transport,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        callback_response = await client.post(
            webhook_url,
            json=response.model_dump(mode="json"),
        )
        callback_response.raise_for_status()


def get_webhook_poster() -> WebhookPoster:
    return post_webhook


async def _run_in_background(
    request: CustomerServiceRequest,
    harness: CustomerServiceHarness,
    webhook_poster: WebhookPoster,
) -> None:
    try:
        response = await harness.run(request)
    except Exception:
        logger.exception(
            "Background harness failed for request_id=%s",
            request.request_id,
        )
        response = CustomerServiceResponse(
            request_id=request.request_id,
            status="error",
            error="Customer service processing failed.",
        )

    webhook_url = request.webhook_url
    if not webhook_url:
        logger.error(
            "Background request missing webhook_url for request_id=%s",
            request.request_id,
        )
        return

    try:
        await webhook_poster(webhook_url, response)
    except Exception:
        logger.exception(
            "Webhook callback failed for request_id=%s",
            request.request_id,
        )


HarnessDependency = Annotated[CustomerServiceHarness, Depends(get_harness)]
WebhookPosterDependency = Annotated[WebhookPoster, Depends(get_webhook_poster)]


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/demo/cases")
async def demo_cases() -> dict[str, list[str]]:
    return {
        "cases": [
            "english_order_lookup",
            "shipping_delay",
            "return_policy",
            "electronics_troubleshooting",
            "pet_product_advice",
            "wig_recommendation",
            "human_requested",
            "high_risk_complaint",
            "german_reply",
            "dynamic_http_tool",
        ]
    }


@router.post("/customer-service/respond")
async def respond(
    request: CustomerServiceRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    harness: HarnessDependency,
    webhook_poster: WebhookPosterDependency,
) -> CustomerServiceResponse | AcceptedResponse:
    if request.async_mode:
        if not request.webhook_url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="webhook_url is required when async_mode is true.",
            )
        try:
            _validate_webhook_url(request.webhook_url)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid webhook_url.",
            ) from exc

        request.request_id = request.request_id or str(uuid4())
        background_tasks.add_task(
            _run_in_background,
            request,
            harness,
            webhook_poster,
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return AcceptedResponse(
            request_id=request.request_id,
            accepted_at=datetime.now(timezone.utc).isoformat(),
        )

    return await harness.run(request)


@router.post("/emailv4")
async def emailv4(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    response: Response,
    harness: HarnessDependency,
    webhook_poster: WebhookPosterDependency,
) -> CustomerServiceResponse | AcceptedResponse:
    try:
        request = convert_emailv4_payload(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Malformed emailv4 payload.",
        ) from exc
    return await respond(
        request,
        background_tasks,
        response,
        harness,
        webhook_poster,
    )
