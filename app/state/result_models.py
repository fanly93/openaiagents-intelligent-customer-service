from typing import Any, Literal

from pydantic import BaseModel, Field


class AcceptedResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    request_id: str
    accepted_at: str
    message: str = "Request accepted; result will be delivered by webhook."


class CustomerServiceResponse(BaseModel):
    request_id: str | None = None
    status: Literal["success", "error"] = "success"
    reply: dict[str, str] = Field(default_factory=dict)
    need_handoff_to_human: bool = False
    handoff_type: Literal["no_handoff", "reply_handoff", "no_reply_handoff"] = "no_handoff"
    handoff_reason: str | None = None
    agent_chain: list[str] = Field(default_factory=lambda: ["customer_service_agent"])
    token_usage: dict[str, int] = Field(default_factory=dict)
    processing_time: float = 0.0
    error: str | None = None
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
