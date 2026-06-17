from typing import Any, Literal

from pydantic import BaseModel, Field

from app.state.request_models import CustomerProfile, ConversationTurn, SlotDefinition


class SlotValue(BaseModel):
    value: Any
    confidence: float = 1.0
    source: Literal["current_message", "history", "tool", "memory"] = "current_message"


class EventSummary(BaseModel):
    name: str
    status: Literal["started", "success", "error"] = "success"
    detail: dict[str, Any] = Field(default_factory=dict)


class CustomerServiceState(BaseModel):
    request_id: str | None = None
    tenant_id: str
    channel: Literal["email", "chat"] = "email"
    subject: str | None = None
    content: str
    customer: CustomerProfile = Field(default_factory=CustomerProfile)
    contexts: list[ConversationTurn] = Field(default_factory=list)

    current_intent: str | None = None
    slot_schema: list[SlotDefinition] = Field(default_factory=list)
    collected_slots: dict[str, SlotValue] = Field(default_factory=dict)
    missing_required_slots: list[str] = Field(default_factory=list)

    selected_template: str | None = None
    template_variables: dict[str, Any] = Field(default_factory=dict)
    retrieved_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    user_memories: list[dict[str, Any]] = Field(default_factory=list)

    order_result: dict[str, Any] | None = None
    logistics_result: dict[str, Any] | None = None
    http_tool_results: dict[str, Any] = Field(default_factory=dict)
    mcp_tool_results: dict[str, Any] = Field(default_factory=dict)
    tool_call_history: list[dict[str, Any]] = Field(default_factory=list)

    need_handoff_to_human: bool = False
    handoff_type: Literal["no_handoff", "reply_handoff", "no_reply_handoff"] = "no_handoff"
    handoff_reason: str | None = None

    final_reply: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    performance_stats: dict[str, Any] = Field(default_factory=dict)
    events: list[EventSummary] = Field(default_factory=list)

    def refresh_missing_required_slots(self) -> None:
        required = [slot.name for slot in self.slot_schema if slot.required]
        self.missing_required_slots = [
            name for name in required if name not in self.collected_slots
        ]

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any], result: Any) -> None:
        self.tool_call_history.append(
            {"tool_name": tool_name, "arguments": arguments, "result": result}
        )
