"""Deterministic local business API mocks.

Run with `.venv/bin/python scripts/run_mock_http_server.py`. The server binds
to 127.0.0.1 and uses port 8765 unless MOCK_HTTP_PORT overrides it.
"""

import os
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints


DEFAULT_MOCK_HTTP_PORT = 8765
MOCK_HTTP_HOST = "127.0.0.1"

app = FastAPI(title="Mock Customer Service Business APIs")


NonBlankOrderId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class OrderRequest(BaseModel):
    order_id: NonBlankOrderId


class OrderData(BaseModel):
    order_id: str
    status: Literal["shipped"]


class OrderResponse(BaseModel):
    code: Literal[200]
    data: OrderData


class LogisticsData(BaseModel):
    order_id: str
    carrier: Literal["DHL"]
    tracking_number: Literal["DHL123456"]
    status: Literal["in_transit"]


class LogisticsResponse(BaseModel):
    code: Literal[200]
    data: LogisticsData


class RefundPolicyData(BaseModel):
    policy: Literal["Returns are accepted within 30 days."]


class RefundPolicyResponse(BaseModel):
    code: Literal[200]
    data: RefundPolicyData


@app.post("/mock/order", response_model=OrderResponse)
async def mock_order(request: OrderRequest) -> OrderResponse:
    return OrderResponse(
        code=200,
        data=OrderData(order_id=request.order_id, status="shipped"),
    )


@app.post("/mock/logistics", response_model=LogisticsResponse)
async def mock_logistics(request: OrderRequest) -> LogisticsResponse:
    return LogisticsResponse(
        code=200,
        data=LogisticsData(
            order_id=request.order_id,
            carrier="DHL",
            tracking_number="DHL123456",
            status="in_transit",
        ),
    )


@app.get("/mock/refund-policy", response_model=RefundPolicyResponse)
async def mock_refund_policy() -> RefundPolicyResponse:
    return RefundPolicyResponse(
        code=200,
        data=RefundPolicyData(
            policy="Returns are accepted within 30 days.",
        ),
    )


def mock_http_port() -> int:
    port = int(os.getenv("MOCK_HTTP_PORT", str(DEFAULT_MOCK_HTTP_PORT)))
    if not 1 <= port <= 65535:
        raise ValueError("MOCK_HTTP_PORT must be between 1 and 65535")
    return port


if __name__ == "__main__":
    uvicorn.run(app, host=MOCK_HTTP_HOST, port=mock_http_port())
