from typing import Any

from app.state.request_models import (
    ConversationTurn,
    CustomerProfile,
    CustomerServiceRequest,
    HttpToolConfig,
    KnowledgeConfig,
    MCPServerConfig,
    MemoryConfig,
    SlotDefinition,
)


def _convert_contexts(raw_contexts: list[dict[str, Any]] | None) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for item in raw_contexts or []:
        content = item.get("content") or item.get("text") or ""
        if content:
            turns.append(ConversationTurn(role=item.get("role", "user"), content=content))
    return turns


def _normalize_channel(value: str | None) -> str:
    if value == "chat":
        return "chat"
    return "email"


def convert_emailv4_payload(payload: dict[str, Any]) -> CustomerServiceRequest:
    customer = CustomerProfile(
        id=payload.get("uuid"),
        name=payload.get("customer_name"),
        email=payload.get("customer_email"),
    )

    return CustomerServiceRequest(
        request_id=payload.get("qid") or payload.get("email_id"),
        tenant_id=str(payload.get("corp") or "default"),
        channel=_normalize_channel(payload.get("channel")),
        subject=payload.get("subject"),
        content=payload["content"],
        customer=customer,
        contexts=_convert_contexts(payload.get("contexts")),
        tools=payload.get("tools") or payload.get("functioncalls") or [],
        slot_schema=[SlotDefinition.model_validate(item) for item in payload.get("slot_schema", [])],
        http_tools=[HttpToolConfig.model_validate(item) for item in payload.get("http_tools", [])],
        mcp_servers=[MCPServerConfig.model_validate(item) for item in payload.get("mcp_servers", [])],
        knowledge_config=KnowledgeConfig.model_validate(payload["knowledge_config"])
        if payload.get("knowledge_config")
        else None,
        memory_config=MemoryConfig.model_validate(payload["memory_config"])
        if payload.get("memory_config")
        else None,
        webhook_url=payload.get("webhook"),
        async_mode=bool(payload.get("async_mode", False)),
        model=payload.get("model") or payload.get("main_model"),
        max_turns=int(payload.get("max_turns") or 6),
        instructions=payload.get("instructions"),
    )
