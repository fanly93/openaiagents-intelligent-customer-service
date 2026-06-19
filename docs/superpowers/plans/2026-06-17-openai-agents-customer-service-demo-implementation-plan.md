# OpenAI Agents Customer Service Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, near-production intelligent customer service Agent backend demo using `openai-agents-python`, FastAPI, configurable tools, mock/remote-ready RAG, MCP, handoff policy, async webhook, and observability.

**Architecture:** Implement a clean Harness Engineering structure: API adapters call `CustomerServiceHarness`, which coordinates state initialization, tool registration, memory/RAG retrieval, OpenAI Agents SDK execution, post-processing, handoff validation, and response reduction. First-version providers use local mock data where appropriate, but all boundaries are designed for later production provider replacement.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, pytest, httpx, uvicorn, `openai-agents-python`, optional Langfuse, JSON mock data, local stdio MCP example server.

---

## Execution Rules

- Work on branch `feature/customer-service-agent-demo`.
- Do not use a separate worktree for this project unless the user later asks for parallel isolated development. Current repository is new and clean, so a feature branch is sufficient.
- Complete the plan in 5 phases.
- After each phase, stop and report to the user before continuing.
- Each phase report must include completed tasks, changed files, verification commands and results, known risks, and the next phase summary.
- Commit at the end of each task or tightly related task group.
- Prefer TDD: write focused tests before implementation where feasible.

## Phase Overview

1. **Phase 1: Project Skeleton and State Models**
2. **Phase 2: Harness Core and OpenAI Agent Execution**
3. **Phase 3: Tools, RAG, Dynamic HTTP, and MCP**
4. **Phase 4: Handoff, Memory, Observability, Async API**
5. **Phase 5: Demo Cases, Tests, Docs, and Final Verification**

---

## File Structure to Create

```text
openaiagents-intelligent-customer-service/
  app/
    __init__.py
    main.py
    config.py
    api/
      __init__.py
      routes.py
      compatibility.py
    harness/
      __init__.py
      customer_service_harness.py
      business_agent_executor.py
      prompt_assembler.py
      reply_post_processor.py
      state_reducer.py
      handoff_policy.py
    observability/
      __init__.py
      hooks.py
      perf.py
      tracing.py
    retrieval/
      __init__.py
      knowledge_retriever.py
      memory_service.py
    state/
      __init__.py
      request_models.py
      conversation_state.py
      result_models.py
    tools/
      __init__.py
      core_tools.py
      dynamic_http_tools.py
      mcp_manager.py
      registry.py
  data/
    knowledge/after_sales.json
    knowledge/product_support.json
    mock_logistics.json
    mock_memories.json
    mock_orders.json
    reply_templates.json
  mcp_servers/
    product_support_server.py
  scripts/
    run_demo_cases.py
    run_mock_http_server.py
  tests/
    conftest.py
    test_api_compatibility.py
    test_demo_cases.py
    test_dynamic_http_tools.py
    test_handoff_policy.py
    test_harness_flow.py
    test_knowledge_retriever.py
    test_reply_post_processor.py
    test_state_models.py
    test_tools_extract_slots.py
  pyproject.toml
  README.md
```

---

## Phase 1: Project Skeleton and State Models

**Phase Goal:** Create a runnable Python project skeleton with typed request, state, and response models plus compatibility mapping for `/emailv4`.

### Task 1.1: Create Python Project Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: command-only validation

- [x] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "openaiagents-intelligent-customer-service"
version = "0.1.0"
description = "Near-production OpenAI Agents intelligent customer service backend demo"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.4.0",
  "httpx>=0.27.0",
  "openai-agents>=0.2.0",
  "mcp>=1.9.0",
  "python-dotenv>=1.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
  "pytest-asyncio>=0.23.0",
  "respx>=0.21.0",
]
observability = [
  "langfuse>=2.45.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]
```

- [x] **Step 2: Create `app/__init__.py`**

```python
"""OpenAI Agents intelligent customer service backend."""
```

- [x] **Step 3: Create `app/config.py`**

```python
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "OpenAI Agents Customer Service"
    environment: Literal["local", "test", "production"] = "local"
    default_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")

    mock_data_dir: str = "data"
    rag_mode: Literal["mock", "remote"] = "mock"
    remote_rag_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [x] **Step 4: Install dependencies**

Run: `.venv/bin/python -m pip install -e ".[dev]"`

Expected: command exits with code 0.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml app/__init__.py app/config.py
git commit -m "chore: initialize python project"
```

### Task 1.2: Define Request Models

**Files:**
- Create: `app/state/__init__.py`
- Create: `app/state/request_models.py`
- Test: `tests/test_state_models.py`

- [x] **Step 1: Write failing tests for request defaults**

Create `tests/test_state_models.py`:

```python
from app.state.request_models import (
    CustomerProfile,
    CustomerServiceRequest,
    SlotDefinition,
)


def test_customer_service_request_defaults_to_email_channel():
    req = CustomerServiceRequest(
        tenant_id="tenant_a",
        content="Where is my order #A100?",
    )

    assert req.channel == "email"
    assert req.async_mode is False
    assert req.max_turns == 6
    assert req.customer == CustomerProfile()


def test_slot_definition_keeps_aliases_and_required_flag():
    slot = SlotDefinition(
        name="product_model",
        description="Product model",
        required=True,
        aliases=["model", "型号"],
    )

    assert slot.name == "product_model"
    assert slot.required is True
    assert slot.aliases == ["model", "型号"]
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_state_models.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing model classes.

- [x] **Step 3: Implement request models**

Create `app/state/__init__.py`:

```python
"""State and API model definitions."""
```

Create `app/state/request_models.py`:

```python
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class CustomerProfile(BaseModel):
    id: str | None = None
    name: str | None = None
    email: str | None = None


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class SlotDefinition(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean", "list"] = "string"
    required: bool = False
    aliases: list[str] = Field(default_factory=list)


class KnowledgeItemConfig(BaseModel):
    kbId: str | list[str] | None = None
    fileIds: list[str] = Field(default_factory=list)
    isAll: bool = False
    kbName: str | None = None


class KnowledgeConfig(BaseModel):
    knowledges: list[KnowledgeItemConfig] = Field(default_factory=list)
    top_k: int = 5
    threshold: float = 0.5


class MemoryConfig(BaseModel):
    enabled: bool = False
    top_k: int = 5


class HttpToolParam(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean"] = "string"
    required: bool = False


class HttpToolConfig(BaseModel):
    name: str
    description: str
    url: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    request_params: list[HttpToolParam] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    response_mapping: dict[str, Any] | None = None
    timeout: float = 10.0
    max_retries: int = 0
    retry_backoff: float = 0.2


class MCPServerConfig(BaseModel):
    name: str
    type: Literal["stdio", "sse", "streamable_http"] = "stdio"
    config: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    tools_filter: dict[str, Any] | None = None
    tools_override: list[dict[str, Any]] = Field(default_factory=list)


class CustomerServiceRequest(BaseModel):
    request_id: str | None = None
    tenant_id: str
    channel: Literal["email", "chat"] = "email"
    subject: str | None = None
    content: str
    customer: CustomerProfile = Field(default_factory=CustomerProfile)
    contexts: list[ConversationTurn] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    slot_schema: list[SlotDefinition] = Field(default_factory=list)
    http_tools: list[HttpToolConfig] = Field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    knowledge_config: KnowledgeConfig | None = None
    memory_config: MemoryConfig | None = None
    webhook_url: str | None = None
    async_mode: bool = False
    model: str | None = None
    max_turns: int = 6
    instructions: str | None = None
```

- [x] **Step 4: Run test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_state_models.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/state/__init__.py app/state/request_models.py tests/test_state_models.py
git commit -m "feat: add customer service request models"
```

### Task 1.3: Define Conversation State and Response Models

**Files:**
- Create: `app/state/conversation_state.py`
- Create: `app/state/result_models.py`
- Modify: `tests/test_state_models.py`

- [x] **Step 1: Add failing tests for state and response**

Append to `tests/test_state_models.py`:

```python
from app.state.conversation_state import CustomerServiceState, SlotValue
from app.state.result_models import CustomerServiceResponse


def test_state_tracks_collected_slots_and_missing_required_slots():
    state = CustomerServiceState(
        request_id="req_1",
        tenant_id="tenant_a",
        channel="email",
        content="My Airdog X5 will not turn on",
        slot_schema=[
            SlotDefinition(name="product_model", description="Product model", required=True),
            SlotDefinition(name="symptom", description="Problem symptom", required=True),
        ],
    )

    state.collected_slots["product_model"] = SlotValue(
        value="Airdog X5",
        confidence=0.9,
        source="current_message",
    )
    state.refresh_missing_required_slots()

    assert state.missing_required_slots == ["symptom"]


def test_response_contains_handoff_fields():
    response = CustomerServiceResponse(
        request_id="req_1",
        status="success",
        reply={"subject": "Re: Support", "body": "We can help."},
        need_handoff_to_human=False,
        handoff_type="no_handoff",
    )

    assert response.reply["body"] == "We can help."
    assert response.need_handoff_to_human is False
```

- [x] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_state_models.py -v`

Expected: FAIL with missing `conversation_state` / `result_models`.

- [x] **Step 3: Implement state model**

Create `app/state/conversation_state.py`:

```python
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
```

- [x] **Step 4: Implement response models**

Create `app/state/result_models.py`:

```python
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
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_state_models.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/state/conversation_state.py app/state/result_models.py tests/test_state_models.py
git commit -m "feat: add conversation state and response models"
```

### Task 1.4: Add `/emailv4` Compatibility Mapping

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/compatibility.py`
- Test: `tests/test_api_compatibility.py`

- [x] **Step 1: Write failing compatibility tests**

Create `tests/test_api_compatibility.py`:

```python
from app.api.compatibility import convert_emailv4_payload


def test_emailv4_payload_maps_to_customer_service_request():
    req = convert_emailv4_payload(
        {
            "corp": "3686",
            "content": "Where is my order #100?",
            "customer_email": "buyer@example.com",
            "customer_name": "Ada",
            "uuid": "user_1",
            "webhook": "https://example.com/hook",
            "contexts": [{"role": "user", "content": "Previous message"}],
        }
    )

    assert req.tenant_id == "3686"
    assert req.customer.email == "buyer@example.com"
    assert req.customer.name == "Ada"
    assert req.customer.id == "user_1"
    assert req.webhook_url == "https://example.com/hook"
    assert req.contexts[0].content == "Previous message"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_api_compatibility.py -v`

Expected: FAIL with missing module.

- [x] **Step 3: Implement compatibility mapper**

Create `app/api/__init__.py`:

```python
"""FastAPI route adapters."""
```

Create `app/api/compatibility.py`:

```python
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


def convert_emailv4_payload(payload: dict[str, Any]) -> CustomerServiceRequest:
    customer = CustomerProfile(
        id=payload.get("uuid"),
        name=payload.get("customer_name"),
        email=payload.get("customer_email"),
    )

    return CustomerServiceRequest(
        request_id=payload.get("qid") or payload.get("email_id"),
        tenant_id=str(payload.get("corp") or "default"),
        channel=payload.get("channel") or "email",
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
```

- [x] **Step 4: Run compatibility tests**

Run: `.venv/bin/python -m pytest tests/test_api_compatibility.py -v`

Expected: PASS.

- [x] **Step 5: Run all Phase 1 tests**

Run: `.venv/bin/python -m pytest tests/test_state_models.py tests/test_api_compatibility.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/api/__init__.py app/api/compatibility.py tests/test_api_compatibility.py
git commit -m "feat: add emailv4 compatibility mapper"
```

### Phase 1 Stop and Report

- [x] Stop after Task 1.4.
- [x] Report changed files, tests run, commits created, and any model/API risks.
- [x] Wait for user approval before Phase 2.

---

## Phase 2: Harness Core and OpenAI Agent Execution

**Phase Goal:** Implement the main harness flow, prompt assembly, OpenAI Agents SDK executor, reply post-processing, and state reduction.

**Phase 2 implementation notes:**

- Preserve the approved single-Agent harness architecture and `tenant_id` request model.
- Pass request-level `model` directly to `BusinessAgentExecutor`; do not add it to persistent conversation state.
- Pass request-level `instructions` into `PromptAssembler` as additive runtime instructions.
- Copy local trace events into `CustomerServiceState.events` before response reduction.
- Target the installed `openai-agents==0.17.5` API: `Agent(...)` accepts tools and MCP servers, while `Runner.run(...)` accepts `context`, `hooks`, and `max_turns`.
- Read aggregate token usage from `result.context_wrapper.usage`; do not infer total usage from only the first and last raw responses.

### Task 2.1: Implement Performance Tracker and Local Trace Skeleton

**Files:**
- Create: `app/observability/__init__.py`
- Create: `app/observability/perf.py`
- Create: `app/observability/tracing.py`
- Test: `tests/test_harness_flow.py`

- [x] **Step 1: Write failing tests for perf and tracing**

Create `tests/test_harness_flow.py`:

```python
from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer


def test_performance_tracker_records_named_operation():
    tracker = PerformanceTracker()
    with tracker.track("prepare_tools"):
        pass

    summary = tracker.summary()
    assert "prepare_tools" in summary
    assert summary["prepare_tools"]["count"] == 1


def test_local_tracer_records_events():
    tracer = LocalTracer(request_id="req_1")
    tracer.record("agent_start", {"model": "gpt-4.1-mini"})

    assert tracer.events[0].name == "agent_start"
    assert tracer.events[0].detail["model"] == "gpt-4.1-mini"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py -v`

Expected: FAIL with missing observability modules.

- [x] **Step 3: Implement perf tracker**

Create `app/observability/__init__.py`:

```python
"""Tracing, hooks, and performance helpers."""
```

Create `app/observability/perf.py`:

```python
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class PerformanceTracker:
    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self._records.setdefault(name, []).append(elapsed)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "total_seconds": round(sum(values), 6),
                "max_seconds": round(max(values), 6),
            }
            for name, values in self._records.items()
        }
```

- [x] **Step 4: Implement local tracer**

Create `app/observability/tracing.py`:

```python
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    name: str
    status: Literal["started", "success", "error"] = "success"
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LocalTracer:
    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        detail: dict[str, Any] | None = None,
        status: Literal["started", "success", "error"] = "success",
    ) -> None:
        self.events.append(TraceEvent(name=name, status=status, detail=detail or {}))
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/observability tests/test_harness_flow.py
git commit -m "feat: add local observability primitives"
```

### Task 2.2: Implement Prompt Assembler and Reply Post Processor

**Files:**
- Create: `app/harness/__init__.py`
- Create: `app/harness/prompt_assembler.py`
- Create: `app/harness/reply_post_processor.py`
- Test: `tests/test_reply_post_processor.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Add failing tests**

Append to `tests/test_harness_flow.py`:

```python
from app.harness.prompt_assembler import PromptAssembler
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import CustomerProfile, SlotDefinition


def test_prompt_assembler_includes_slot_schema_and_language_rules():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="My device has E01",
        customer=CustomerProfile(name="Ada"),
        slot_schema=[SlotDefinition(name="error_code", description="Device error code")],
    )

    instructions = PromptAssembler().build_instructions(
        state,
        tool_guide="tools here",
        extra_instructions="Follow tenant refund policy.",
    )

    assert "error_code" in instructions
    assert "Use the customer's language" in instructions
    assert "tools here" in instructions
    assert "Follow tenant refund policy." in instructions
```

Create `tests/test_reply_post_processor.py`:

```python
from app.harness.reply_post_processor import ReplyPostProcessor


def test_reply_post_processor_removes_react_markers():
    processor = ReplyPostProcessor()
    cleaned = processor.clean(
        "THINK: I need a tool\nACTION: call tool\nDear customer,\nWe can help."
    )

    assert "THINK:" not in cleaned
    assert "ACTION:" not in cleaned
    assert "Dear customer" in cleaned
```

- [x] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py tests/test_reply_post_processor.py -v`

Expected: FAIL with missing harness modules.

- [x] **Step 3: Implement harness package and prompt assembler**

Create `app/harness/__init__.py`:

```python
"""Harness components for customer service Agent execution."""
```

Create `app/harness/prompt_assembler.py`:

```python
from app.state.conversation_state import CustomerServiceState


class PromptAssembler:
    def build_instructions(
        self,
        state: CustomerServiceState,
        tool_guide: str = "",
        extra_instructions: str | None = None,
    ) -> str:
        slot_lines = [
            f"- {slot.name}: {slot.description} (required={slot.required})"
            for slot in state.slot_schema
        ]
        memory_lines = [
            f"- {memory.get('memory', memory)}"
            for memory in state.user_memories
        ]
        knowledge_lines = [
            f"- {item.get('text', item)}"
            for item in state.retrieved_knowledge
        ]

        return "\n\n".join(
            part
            for part in [
                "You are a professional customer service Agent for global ecommerce brands.",
                "Use the customer's language consistently across greeting, body, and signature.",
                "Extract configured slots before calling business tools when possible.",
                "Call handoff_to_human when the request needs human support.",
                "Never output THINK, ACTION, OBSERVE, or internal reasoning markers.",
                "## Slot Schema\n" + "\n".join(slot_lines) if slot_lines else "",
                "## User Memories\n" + "\n".join(memory_lines) if memory_lines else "",
                "## Retrieved Knowledge\n" + "\n".join(knowledge_lines) if knowledge_lines else "",
                "## Available Tools\n" + tool_guide if tool_guide else "",
                "## Tenant Instructions\n" + extra_instructions if extra_instructions else "",
            ]
            if part
        )

    def build_user_message(self, state: CustomerServiceState) -> str:
        history = "\n".join(f"{turn.role}: {turn.content}" for turn in state.contexts)
        if history:
            return f"Current message:\n{state.content}\n\nConversation history:\n{history}"
        return state.content
```

- [x] **Step 4: Implement reply post processor**

Create `app/harness/reply_post_processor.py`:

```python
import re


class ReplyPostProcessor:
    _line_patterns = [
        re.compile(r"^\s*THINK:.*$", re.IGNORECASE),
        re.compile(r"^\s*ACTION:.*$", re.IGNORECASE),
        re.compile(r"^\s*OBSERVE:.*$", re.IGNORECASE),
        re.compile(r"^\s*OBSERVATION:.*$", re.IGNORECASE),
    ]

    def clean(self, content: str | None) -> str:
        if not content:
            return ""

        lines: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if any(pattern.match(stripped) for pattern in self._line_patterns):
                continue
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        return re.sub(r"\n{3,}", "\n\n", cleaned)
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py tests/test_reply_post_processor.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/harness tests/test_harness_flow.py tests/test_reply_post_processor.py
git commit -m "feat: add prompt assembly and reply post processing"
```

### Task 2.3: Implement State Reducer and Agent Executor Interface

**Files:**
- Create: `app/harness/state_reducer.py`
- Create: `app/harness/business_agent_executor.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Add failing tests**

Append to `tests/test_harness_flow.py`:

```python
import pytest

from app.harness.business_agent_executor import AgentRunResult, BusinessAgentExecutor
from app.harness.state_reducer import StateReducer


def test_state_reducer_builds_response_from_state():
    state = CustomerServiceState(
        request_id="req_1",
        tenant_id="tenant_a",
        channel="email",
        subject="Order status",
        content="Where is my order?",
        final_reply="Your order has shipped.",
    )

    response = StateReducer().build_response(state, processing_time=1.25)

    assert response.reply["body"] == "Your order has shipped."
    assert response.processing_time == 1.25


@pytest.mark.asyncio
async def test_business_agent_executor_allows_mock_runner():
    async def fake_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        assert model == "gpt-test"
        return AgentRunResult(final_output="Hello from mock", token_usage={"total_tokens": 10})

    executor = BusinessAgentExecutor(runner=fake_runner)
    state = CustomerServiceState(tenant_id="tenant_a", channel="chat", content="Hi")
    result = await executor.run(
        state=state,
        instructions="instructions",
        user_message="Hi",
        tools=[],
        mcp_servers=[],
        max_turns=3,
        model="gpt-test",
    )

    assert result.final_output == "Hello from mock"
    assert result.token_usage["total_tokens"] == 10
```

- [x] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py -v`

Expected: FAIL with missing modules/classes.

- [x] **Step 3: Implement state reducer**

Create `app/harness/state_reducer.py`:

```python
from app.state.conversation_state import CustomerServiceState
from app.state.result_models import CustomerServiceResponse


class StateReducer:
    def build_response(
        self,
        state: CustomerServiceState,
        processing_time: float,
        error: str | None = None,
    ) -> CustomerServiceResponse:
        body = "" if state.handoff_type == "no_reply_handoff" else (state.final_reply or "")
        reply = {"subject": state.subject or "Customer support", "body": body}
        return CustomerServiceResponse(
            request_id=state.request_id,
            status="error" if error else "success",
            reply=reply,
            need_handoff_to_human=state.need_handoff_to_human,
            handoff_type=state.handoff_type,
            handoff_reason=state.handoff_reason,
            token_usage=state.token_usage,
            processing_time=processing_time,
            error=error,
            state_snapshot=state.model_dump(mode="json"),
        )
```

- [x] **Step 4: Implement business agent executor**

Create `app/harness/business_agent_executor.py`:

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.state.conversation_state import CustomerServiceState


@dataclass
class AgentRunResult:
    final_output: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_responses: list[Any] = field(default_factory=list)


RunnerCallable = Callable[
    [CustomerServiceState, str, str, list[Any], list[Any], int, str | None],
    Awaitable[AgentRunResult],
]


class BusinessAgentExecutor:
    def __init__(self, runner: RunnerCallable | None = None) -> None:
        self._runner = runner

    async def run(
        self,
        state: CustomerServiceState,
        instructions: str,
        user_message: str,
        tools: list[Any],
        mcp_servers: list[Any],
        max_turns: int,
        model: str | None = None,
    ) -> AgentRunResult:
        if self._runner:
            return await self._runner(
                state,
                instructions,
                user_message,
                tools,
                mcp_servers,
                max_turns,
                model,
            )
        return await self._run_openai_agents(
            state,
            instructions,
            user_message,
            tools,
            mcp_servers,
            max_turns,
            model,
        )

    async def _run_openai_agents(
        self,
        state: CustomerServiceState,
        instructions: str,
        user_message: str,
        tools: list[Any],
        mcp_servers: list[Any],
        max_turns: int,
        model: str | None,
    ) -> AgentRunResult:
        from agents import Agent, Runner

        settings = get_settings()
        agent = Agent[CustomerServiceState](
            name="Customer Service Agent",
            instructions=instructions,
            model=model or settings.default_model,
            tools=tools,
            mcp_servers=mcp_servers,
        )
        result = await Runner.run(
            starting_agent=agent,
            input=user_message,
            context=state,
            max_turns=max_turns,
        )
        raw_responses = getattr(result, "raw_responses", []) or []
        aggregate_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        usage = (
            {
                "input_tokens": aggregate_usage.input_tokens,
                "output_tokens": aggregate_usage.output_tokens,
                "total_tokens": aggregate_usage.total_tokens,
            }
            if aggregate_usage
            else {}
        )
        return AgentRunResult(
            final_output=str(getattr(result, "final_output", result)),
            token_usage=usage,
            raw_responses=raw_responses,
        )
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/harness/state_reducer.py app/harness/business_agent_executor.py tests/test_harness_flow.py
git commit -m "feat: add state reducer and agent executor"
```

### Task 2.4: Implement CustomerServiceHarness with Mock Executor Support

**Files:**
- Create: `app/harness/customer_service_harness.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Add failing end-to-end harness test**

Append to `tests/test_harness_flow.py`:

```python
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.request_models import CustomerServiceRequest


@pytest.mark.asyncio
async def test_customer_service_harness_runs_with_mock_executor():
    async def fake_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        assert "Use tenant-specific warranty wording." in instructions
        assert model == "gpt-test"
        return AgentRunResult(final_output="Dear Ada,\nYour order has shipped.")

    harness = CustomerServiceHarness(executor=BusinessAgentExecutor(runner=fake_runner))
    response = await harness.run(
        CustomerServiceRequest(
            request_id="req_1",
            tenant_id="tenant_a",
            channel="email",
            subject="Order",
            content="Where is my order?",
            customer=CustomerProfile(name="Ada"),
            model="gpt-test",
            instructions="Use tenant-specific warranty wording.",
        )
    )

    assert response.status == "success"
    assert "Your order has shipped" in response.reply["body"]
    assert response.state_snapshot["events"][0]["name"] == "request_start"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_customer_service_harness_runs_with_mock_executor -v`

Expected: FAIL with missing `CustomerServiceHarness`.

- [x] **Step 3: Implement harness**

Create `app/harness/customer_service_harness.py`:

```python
from time import perf_counter

from app.harness.business_agent_executor import BusinessAgentExecutor
from app.harness.prompt_assembler import PromptAssembler
from app.harness.reply_post_processor import ReplyPostProcessor
from app.harness.state_reducer import StateReducer
from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer
from app.state.conversation_state import CustomerServiceState, EventSummary
from app.state.request_models import CustomerServiceRequest
from app.state.result_models import CustomerServiceResponse


class CustomerServiceHarness:
    def __init__(
        self,
        executor: BusinessAgentExecutor | None = None,
        prompt_assembler: PromptAssembler | None = None,
        reply_processor: ReplyPostProcessor | None = None,
        state_reducer: StateReducer | None = None,
    ) -> None:
        self.executor = executor or BusinessAgentExecutor()
        self.prompt_assembler = prompt_assembler or PromptAssembler()
        self.reply_processor = reply_processor or ReplyPostProcessor()
        self.state_reducer = state_reducer or StateReducer()

    async def run(self, request: CustomerServiceRequest) -> CustomerServiceResponse:
        start = perf_counter()
        tracker = PerformanceTracker()
        tracer = LocalTracer(request_id=request.request_id)
        state = self._init_state(request)

        try:
            tracer.record("request_start", {"tenant_id": request.tenant_id, "channel": request.channel})
            with tracker.track("assemble_prompt"):
                instructions = self.prompt_assembler.build_instructions(
                    state,
                    extra_instructions=request.instructions,
                )
                user_message = self.prompt_assembler.build_user_message(state)
            with tracker.track("agent_run"):
                result = await self.executor.run(
                    state=state,
                    instructions=instructions,
                    user_message=user_message,
                    tools=[],
                    mcp_servers=[],
                    max_turns=request.max_turns,
                    model=request.model,
                )
            state.final_reply = self.reply_processor.clean(result.final_output)
            state.token_usage = result.token_usage
            state.performance_stats = tracker.summary()
            tracer.record("response_built", {"reply_length": len(state.final_reply or "")})
            self._sync_trace_events(state, tracer)
            processing_time = perf_counter() - start
            return self.state_reducer.build_response(state, processing_time=round(processing_time, 6))
        except Exception as exc:
            tracer.record("request_error", {"error": str(exc)}, status="error")
            self._sync_trace_events(state, tracer)
            processing_time = perf_counter() - start
            return self.state_reducer.build_response(state, processing_time=round(processing_time, 6), error=str(exc))

    def _init_state(self, request: CustomerServiceRequest) -> CustomerServiceState:
        return CustomerServiceState(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            channel=request.channel,
            subject=request.subject,
            content=request.content,
            customer=request.customer,
            contexts=request.contexts,
            slot_schema=request.slot_schema,
        )

    @staticmethod
    def _sync_trace_events(
        state: CustomerServiceState,
        tracer: LocalTracer,
    ) -> None:
        state.events = [
            EventSummary(name=event.name, status=event.status, detail=event.detail)
            for event in tracer.events
        ]
```

- [x] **Step 4: Run harness tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/harness/customer_service_harness.py tests/test_harness_flow.py
git commit -m "feat: add customer service harness skeleton"
```

### Phase 2 Stop and Report

- [x] Stop after Task 2.4.
- [x] Report changed files, tests run, commits created, and any Agent SDK integration risks.
- [x] Wait for user approval before Phase 3.

---

## Phase 3: Tools, RAG, Dynamic HTTP, and MCP

**Phase Goal:** Add the core tool layer, configurable slot extraction, mock/remote-ready RAG, dynamic HTTP tools, order/logistics mock tools, and a local MCP example server.

### Task 3.1: Implement `extract_slots`

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/core_tools.py`
- Test: `tests/test_tools_extract_slots.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_tools_extract_slots.py`:

```python
import pytest

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import SlotDefinition
from app.tools.core_tools import extract_slots_impl


@pytest.mark.asyncio
async def test_extract_slots_writes_configured_slots_only():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="Airdog X5 has E01",
        slot_schema=[
            SlotDefinition(name="product_model", description="Product model", required=True),
            SlotDefinition(name="error_code", description="Error code", required=False),
        ],
    )

    result = await extract_slots_impl(
        state,
        [
            {"name": "product_model", "value": "Airdog X5", "confidence": 0.9, "source": "current_message"},
            {"name": "unknown", "value": "ignored", "confidence": 0.9, "source": "current_message"},
        ],
    )

    assert result["accepted"] == ["product_model"]
    assert result["ignored"] == ["unknown"]
    assert state.collected_slots["product_model"].value == "Airdog X5"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_tools_extract_slots.py -v`

Expected: FAIL with missing tools module.

- [x] **Step 3: Implement core tool logic**

Create `app/tools/__init__.py`:

```python
"""Agent tools and tool registry."""
```

Create `app/tools/core_tools.py`:

```python
from typing import Any

from app.state.conversation_state import CustomerServiceState, SlotValue


async def extract_slots_impl(
    state: CustomerServiceState,
    slots: list[dict[str, Any]],
) -> dict[str, list[str]]:
    schema_names = {slot.name for slot in state.slot_schema}
    accepted: list[str] = []
    ignored: list[str] = []

    for item in slots:
        name = str(item.get("name", "")).strip()
        if name not in schema_names:
            ignored.append(name)
            continue
        state.collected_slots[name] = SlotValue(
            value=item.get("value"),
            confidence=float(item.get("confidence", 1.0)),
            source=item.get("source", "current_message"),
        )
        accepted.append(name)

    state.refresh_missing_required_slots()
    state.record_tool_call("extract_slots", {"slots": slots}, {"accepted": accepted, "ignored": ignored})
    return {"accepted": accepted, "ignored": ignored}
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tools_extract_slots.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/tools tests/test_tools_extract_slots.py
git commit -m "feat: add configurable slot extraction"
```

### Task 3.2: Implement Mock Knowledge Retriever

**Files:**
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/knowledge_retriever.py`
- Create: `data/knowledge/after_sales.json`
- Create: `data/knowledge/product_support.json`
- Test: `tests/test_knowledge_retriever.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_knowledge_retriever.py`:

```python
import pytest

from app.retrieval.knowledge_retriever import MockKnowledgeRetriever
from app.state.request_models import KnowledgeConfig, KnowledgeItemConfig


@pytest.mark.asyncio
async def test_mock_knowledge_retriever_filters_by_file_ids():
    retriever = MockKnowledgeRetriever(data_dir="data/knowledge")
    result = await retriever.retrieve(
        tenant_id="tenant_a",
        query="return policy",
        knowledge_config=KnowledgeConfig(
            knowledges=[
                KnowledgeItemConfig(kbId="kb_after_sales", fileIds=["return_policy"])
            ],
            top_k=5,
            threshold=0.1,
        ),
    )

    assert result
    assert all(item["file_id"] == "return_policy" for item in result)
```

- [x] **Step 2: Create mock data**

Create `data/knowledge/after_sales.json`:

```json
[
  {
    "tenant_id": "tenant_a",
    "kb_id": "kb_after_sales",
    "file_id": "return_policy",
    "text": "Customers can request a return within 30 days if the item is unused and in original packaging.",
    "keywords": ["return", "refund", "policy"]
  },
  {
    "tenant_id": "tenant_a",
    "kb_id": "kb_after_sales",
    "file_id": "warranty_policy",
    "text": "Most electronic products include a one-year limited warranty for manufacturing defects.",
    "keywords": ["warranty", "defect", "repair"]
  }
]
```

Create `data/knowledge/product_support.json`:

```json
[
  {
    "tenant_id": "tenant_a",
    "kb_id": "kb_product_support",
    "file_id": "troubleshooting_3c",
    "text": "For E01 errors, restart the device, check the filter compartment, and confirm the power adapter is connected.",
    "keywords": ["E01", "restart", "power", "filter"]
  }
]
```

- [x] **Step 3: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_knowledge_retriever.py -v`

Expected: FAIL with missing retriever.

- [x] **Step 4: Implement retriever**

Create `app/retrieval/__init__.py`:

```python
"""Knowledge and memory providers."""
```

Create `app/retrieval/knowledge_retriever.py`:

```python
import json
from pathlib import Path
from typing import Any

from app.state.request_models import KnowledgeConfig


class MockKnowledgeRetriever:
    def __init__(self, data_dir: str = "data/knowledge") -> None:
        self.data_dir = Path(data_dir)

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        knowledge_config: KnowledgeConfig | None = None,
    ) -> list[dict[str, Any]]:
        docs = self._load_docs()
        kb_ids, file_ids, top_k, threshold = self._parse_config(knowledge_config)
        query_terms = {term.lower() for term in query.replace("#", " ").split() if term}

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            if doc.get("tenant_id") != tenant_id:
                continue
            if kb_ids and doc.get("kb_id") not in kb_ids:
                continue
            if file_ids and doc.get("file_id") not in file_ids:
                continue
            keywords = {str(item).lower() for item in doc.get("keywords", [])}
            text_terms = {term.lower().strip(".,!?") for term in doc.get("text", "").split()}
            overlap = query_terms & (keywords | text_terms)
            score = len(overlap) / max(len(query_terms), 1)
            if score >= threshold:
                item = dict(doc)
                item["score"] = round(score, 4)
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _load_docs(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("*.json")):
            docs.extend(json.loads(path.read_text(encoding="utf-8")))
        return docs

    def _parse_config(self, config: KnowledgeConfig | None) -> tuple[set[str], set[str], int, float]:
        if not config:
            return set(), set(), 5, 0.0
        kb_ids: set[str] = set()
        file_ids: set[str] = set()
        for item in config.knowledges:
            if isinstance(item.kbId, str):
                kb_ids.add(item.kbId)
            elif isinstance(item.kbId, list):
                kb_ids.update(str(kb_id) for kb_id in item.kbId)
            file_ids.update(item.fileIds)
        return kb_ids, file_ids, config.top_k, config.threshold
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_knowledge_retriever.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/retrieval data/knowledge tests/test_knowledge_retriever.py
git commit -m "feat: add mock knowledge retriever"
```

### Task 3.3: Implement Dynamic HTTP Tool Builder

**Files:**
- Create: `app/tools/dynamic_http_tools.py`
- Test: `tests/test_dynamic_http_tools.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_dynamic_http_tools.py`:

```python
import respx
from httpx import Response
import pytest

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import HttpToolConfig, HttpToolParam
from app.tools.dynamic_http_tools import call_dynamic_http_tool


@pytest.mark.asyncio
@respx.mock
async def test_dynamic_http_tool_calls_post_and_maps_response():
    respx.post("https://mock.local/order").mock(
        return_value=Response(200, json={"code": 200, "data": {"status": "shipped"}})
    )
    state = CustomerServiceState(tenant_id="tenant_a", channel="email", content="Order")
    config = HttpToolConfig(
        name="get_order",
        description="Get order",
        url="https://mock.local/order",
        method="POST",
        request_params=[HttpToolParam(name="order_id", description="Order id", required=True)],
    )

    result = await call_dynamic_http_tool(state, config, {"order_id": "A100"})

    assert result == {"status": "shipped"}
    assert state.http_tool_results
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dynamic_http_tools.py -v`

Expected: FAIL with missing dynamic HTTP module.

- [x] **Step 3: Implement dynamic HTTP call helper**

Create `app/tools/dynamic_http_tools.py`:

```python
import asyncio
from typing import Any

import httpx

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import HttpToolConfig


def _cache_key(tool_name: str, params: dict[str, Any]) -> str:
    pairs = ",".join(f"{key}={params[key]}" for key in sorted(params))
    return f"{tool_name}:{pairs}"


async def call_dynamic_http_tool(
    state: CustomerServiceState,
    config: HttpToolConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    missing = [
        param.name
        for param in config.request_params
        if param.required and not params.get(param.name)
    ]
    if missing:
        return {"error": "missing_required_params", "missing": missing}

    key = _cache_key(config.name, params)
    if key in state.http_tool_results:
        return state.http_tool_results[key]

    last_error: str | None = None
    for attempt in range(config.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                if config.method == "GET":
                    response = await client.get(config.url, params=params, headers=config.headers)
                elif config.method == "POST":
                    response = await client.post(config.url, json=params, headers=config.headers)
                elif config.method == "PUT":
                    response = await client.put(config.url, json=params, headers=config.headers)
                elif config.method == "DELETE":
                    response = await client.delete(config.url, params=params, headers=config.headers)
                else:
                    response = await client.patch(config.url, json=params, headers=config.headers)
                response.raise_for_status()
                data = response.json()
                mapped = _map_response(data, config.response_mapping)
                state.http_tool_results[key] = mapped
                state.record_tool_call(config.name, params, mapped)
                return mapped
        except Exception as exc:
            last_error = str(exc)
            if attempt < config.max_retries:
                await asyncio.sleep(config.retry_backoff * (2 ** attempt))

    result = {"error": "http_tool_failed", "message": last_error or "unknown error"}
    state.record_tool_call(config.name, params, result)
    return result


def _map_response(data: dict[str, Any], mapping: dict[str, Any] | None) -> dict[str, Any]:
    success_field = (mapping or {}).get("success_field", "code")
    success_value = (mapping or {}).get("success_value", 200)
    data_field = (mapping or {}).get("data_field", "data")
    if str(data.get(success_field)) == str(success_value):
        value = data.get(data_field, {})
        return value if isinstance(value, dict) else {"value": value}
    return {"error": "api_error", "raw": data}
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_dynamic_http_tools.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/tools/dynamic_http_tools.py tests/test_dynamic_http_tools.py
git commit -m "feat: add dynamic http tool executor"
```

### Task 3.4: Implement Tool Registry, Mock Order/Logistics, and MCP Manager Stub

**Files:**
- Create: `app/tools/registry.py`
- Create: `app/tools/mcp_manager.py`
- Create: `data/mock_orders.json`
- Create: `data/mock_logistics.json`
- Create: `mcp_servers/product_support_server.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Create mock order and logistics data**

Create `data/mock_orders.json`:

```json
{
  "A100": {
    "order_id": "A100",
    "status": "shipped",
    "items": ["Airdog X5"],
    "total": 299.0
  },
  "WIG200": {
    "order_id": "WIG200",
    "status": "processing",
    "items": ["Body Wave Wig"],
    "total": 129.0
  }
}
```

Create `data/mock_logistics.json`:

```json
{
  "A100": {
    "order_id": "A100",
    "carrier": "DHL",
    "tracking_number": "DHL123456",
    "status": "in_transit"
  }
}
```

- [x] **Step 2: Implement registry**

Create `app/tools/registry.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from app.state.request_models import CustomerServiceRequest


@dataclass
class ToolSetup:
    tools: list[Any] = field(default_factory=list)
    tool_guide: str = ""
    enabled_names: list[str] = field(default_factory=list)


class ToolRegistry:
    async def prepare(self, request: CustomerServiceRequest) -> ToolSetup:
        enabled = ["extract_slots", "get_rag_knowledge", "handoff_to_human"]
        enabled.extend(name for name in request.tools if name not in enabled)
        guide = "\n".join(f"- {name}" for name in enabled)
        return ToolSetup(tools=[], tool_guide=guide, enabled_names=enabled)
```

- [x] **Step 3: Implement MCP manager stub**

Create `app/tools/mcp_manager.py`:

```python
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from app.state.request_models import MCPServerConfig


@dataclass
class MCPSetup:
    servers: list[Any] = field(default_factory=list)
    enabled_names: list[str] = field(default_factory=list)
    guide_text: str = ""


class MCPManager:
    async def prepare(self, configs: list[MCPServerConfig]) -> MCPSetup:
        names = [config.name for config in configs]
        guide = "\n".join(f"- {name}" for name in names)
        return MCPSetup(servers=[], enabled_names=names, guide_text=guide)

    def exit_stack(self) -> AsyncExitStack:
        return AsyncExitStack()
```

- [x] **Step 4: Create local MCP example server file**

Create `mcp_servers/product_support_server.py`:

```python
"""Local stdio MCP example server reference for product support tools.

The first implementation phase keeps this file as an executable reference target.
The MCPManager can later launch it through a stdio MCP config.
"""


def lookup_product_manual(product_or_sku: str, question: str) -> str:
    return f"Manual guidance for {product_or_sku}: restart the device and check the quick-start guide."


def check_warranty_policy(product_or_sku: str) -> str:
    return f"{product_or_sku} includes a one-year limited warranty for manufacturing defects."
```

- [x] **Step 5: Add registry test**

Append to `tests/test_harness_flow.py`:

```python
from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_tool_registry_includes_core_tools():
    setup = await ToolRegistry().prepare(
        CustomerServiceRequest(tenant_id="tenant_a", content="Help")
    )

    assert "extract_slots" in setup.enabled_names
    assert "get_rag_knowledge" in setup.enabled_names
    assert "handoff_to_human" in setup.enabled_names
```

- [x] **Step 6: Run Phase 3 tests**

Run: `.venv/bin/python -m pytest tests/test_tools_extract_slots.py tests/test_knowledge_retriever.py tests/test_dynamic_http_tools.py tests/test_harness_flow.py -v`

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add app/tools data/mock_orders.json data/mock_logistics.json mcp_servers tests/test_harness_flow.py
git commit -m "feat: add tool registry and mcp scaffold"
```

### Task 3.5: Upgrade MCP Example to a Runnable stdio Server

**Files:**
- Modify: `mcp_servers/product_support_server.py`
- Modify: `app/tools/mcp_manager.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Add MCP manager config test**

Append to `tests/test_harness_flow.py`:

```python
from app.state.request_models import MCPServerConfig
from app.tools.mcp_manager import MCPManager


@pytest.mark.asyncio
async def test_mcp_manager_accepts_stdio_config():
    setup = await MCPManager().prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="stdio",
                config={
                    "command": "python",
                    "args": ["mcp_servers/product_support_server.py"],
                },
            )
        ]
    )

    assert setup.enabled_names == ["product_support"]
    assert "product_support" in setup.guide_text
```

- [x] **Step 2: Replace example server with FastMCP server**

Update `mcp_servers/product_support_server.py`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("product_support")


@mcp.tool()
def lookup_product_manual(product_or_sku: str, question: str) -> str:
    """Look up product manual guidance for a product or SKU."""
    if "e01" in question.lower():
        return (
            f"For {product_or_sku}, E01 usually means the device needs a restart, "
            "a filter compartment check, and a power adapter check."
        )
    return f"Manual guidance for {product_or_sku}: restart the device and check the quick-start guide."


@mcp.tool()
def check_warranty_policy(product_or_sku: str) -> str:
    """Return warranty policy for a product or SKU."""
    return f"{product_or_sku} includes a one-year limited warranty for manufacturing defects."


if __name__ == "__main__":
    mcp.run()
```

- [x] **Step 3: Implement MCPManager using OpenAI Agents MCP classes**

Update `app/tools/mcp_manager.py`:

```python
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from app.state.request_models import MCPServerConfig


@dataclass
class MCPSetup:
    servers: list[Any] = field(default_factory=list)
    enabled_names: list[str] = field(default_factory=list)
    guide_text: str = ""


class MCPManager:
    async def prepare(self, configs: list[MCPServerConfig]) -> MCPSetup:
        from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

        servers: list[Any] = []
        enabled_names: list[str] = []
        guide_lines: list[str] = []

        for config in configs:
            if config.type == "stdio":
                command = config.config.get("command")
                if not command:
                    continue
                server = MCPServerStdio(
                    name=config.name,
                    command=command,
                    args=config.config.get("args", []),
                    env=config.config.get("env", {}),
                )
                guide_lines.append(f"- {config.name} (stdio): {command}")
            elif config.type == "sse":
                url = config.config.get("url")
                if not url:
                    continue
                server = MCPServerSse(
                    name=config.name,
                    params={"url": url, "headers": config.headers},
                    client_session_timeout_seconds=config.config.get("timeout", 30),
                )
                guide_lines.append(f"- {config.name} (sse): {url}")
            else:
                url = config.config.get("url")
                if not url:
                    continue
                server = MCPServerStreamableHttp(
                    name=config.name,
                    params={"url": url, "headers": config.headers},
                    client_session_timeout_seconds=config.config.get("timeout", 30),
                )
                guide_lines.append(f"- {config.name} (streamable_http): {url}")

            server._tools_filter = config.tools_filter
            server._tools_override = config.tools_override
            servers.append(server)
            enabled_names.append(config.name)

        return MCPSetup(
            servers=servers,
            enabled_names=enabled_names,
            guide_text="\n".join(guide_lines),
        )

    def exit_stack(self) -> AsyncExitStack:
        return AsyncExitStack()
```

- [x] **Step 4: Run MCP-related tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_mcp_manager_accepts_stdio_config -v`

Expected: PASS.

- [x] **Step 5: Syntax check MCP server**

Run: `.venv/bin/python -m py_compile mcp_servers/product_support_server.py`

Expected: command exits with code 0.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml app/tools/mcp_manager.py mcp_servers/product_support_server.py tests/test_harness_flow.py
git commit -m "feat: add runnable local mcp server"
```

### Phase 3 Stop and Report

- [x] Stop after Task 3.5.
- [x] Report changed files, tests run, commits created, and remaining tool/MCP limitations.
- [x] Wait for user approval before Phase 4.

---

## Phase 4: Handoff, Memory, Observability, Async API

**Phase Goal:** Add handoff policy, mock memory, Langfuse/local tracing integration hooks, FastAPI routes, async response mode, and webhook callback behavior.

### Task 4.1: Implement Handoff Policy

**Files:**
- Create: `app/harness/handoff_policy.py`
- Test: `tests/test_handoff_policy.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_handoff_policy.py`:

```python
from app.harness.handoff_policy import HandoffPolicy
from app.state.conversation_state import CustomerServiceState


def test_handoff_policy_marks_empty_reply_for_handoff():
    state = CustomerServiceState(tenant_id="tenant_a", channel="email", content="Help")
    state.final_reply = ""

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "reply_handoff"


def test_handoff_policy_corrects_inconsistent_state():
    state = CustomerServiceState(tenant_id="tenant_a", channel="email", content="Help")
    state.need_handoff_to_human = True
    state.handoff_type = "no_handoff"

    HandoffPolicy().apply(state)

    assert state.handoff_type == "reply_handoff"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_handoff_policy.py -v`

Expected: FAIL with missing policy.

- [x] **Step 3: Implement policy**

Create `app/harness/handoff_policy.py`:

```python
from app.state.conversation_state import CustomerServiceState


class HandoffPolicy:
    high_risk_terms = ["lawyer", "legal", "lawsuit", "fraud", "scam", "angry", "complaint"]
    human_terms = ["human", "agent", "representative", "人工", "客服"]

    def apply(self, state: CustomerServiceState) -> None:
        content = state.content.lower()

        if not state.final_reply or len(state.final_reply.strip()) < 10:
            self._mark_reply_handoff(state, "reply is empty or too short")

        if any(term in content for term in self.high_risk_terms):
            state.need_handoff_to_human = True
            state.handoff_type = "no_reply_handoff"
            state.handoff_reason = "high risk complaint or escalation"

        elif any(term in content for term in self.human_terms):
            self._mark_reply_handoff(state, "customer requested human support")

        if state.need_handoff_to_human and state.handoff_type == "no_handoff":
            self._mark_reply_handoff(state, "corrected inconsistent handoff state")

        if not state.need_handoff_to_human and state.handoff_type != "no_handoff":
            state.handoff_type = "no_handoff"
            state.handoff_reason = None

    def _mark_reply_handoff(self, state: CustomerServiceState, reason: str) -> None:
        state.need_handoff_to_human = True
        if state.handoff_type == "no_handoff":
            state.handoff_type = "reply_handoff"
        state.handoff_reason = state.handoff_reason or reason
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_handoff_policy.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/harness/handoff_policy.py tests/test_handoff_policy.py
git commit -m "feat: add handoff policy"
```

### Task 4.2: Implement Mock Memory Service

**Files:**
- Create: `app/retrieval/memory_service.py`
- Create: `data/mock_memories.json`
- Test: `tests/test_harness_flow.py`

- [x] **Step 1: Add memory data**

Create `data/mock_memories.json`:

```json
{
  "user_ada": [
    {
      "memory": "Customer prefers German responses when available.",
      "keywords": ["german", "deutsch", "language"]
    },
    {
      "memory": "Customer previously reported an E01 error on Airdog X5.",
      "keywords": ["E01", "Airdog", "X5"]
    }
  ]
}
```

- [x] **Step 2: Add failing memory test**

Append to `tests/test_harness_flow.py`:

```python
from app.retrieval.memory_service import MockMemoryService


@pytest.mark.asyncio
async def test_mock_memory_service_retrieves_user_memory():
    memories = await MockMemoryService(data_path="data/mock_memories.json").retrieve(
        customer_id="user_ada",
        query="Airdog X5 E01",
        top_k=3,
    )

    assert memories
    assert "E01" in memories[0]["memory"]
```

- [x] **Step 3: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_mock_memory_service_retrieves_user_memory -v`

Expected: FAIL with missing memory service.

- [x] **Step 4: Implement memory service**

Create `app/retrieval/memory_service.py`:

```python
import json
from pathlib import Path


class MockMemoryService:
    def __init__(self, data_path: str = "data/mock_memories.json") -> None:
        self.data_path = Path(data_path)

    async def retrieve(self, customer_id: str | None, query: str, top_k: int = 5) -> list[dict[str, object]]:
        if not customer_id or not self.data_path.exists():
            return []
        data = json.loads(self.data_path.read_text(encoding="utf-8"))
        memories = data.get(customer_id, [])
        query_terms = {term.lower().strip(".,!?") for term in query.split()}
        scored: list[tuple[int, dict[str, object]]] = []
        for memory in memories:
            keywords = {str(item).lower() for item in memory.get("keywords", [])}
            score = len(query_terms & keywords)
            if score:
                item = dict(memory)
                item["score"] = score
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]
```

- [x] **Step 5: Run test**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_mock_memory_service_retrieves_user_memory -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/retrieval/memory_service.py data/mock_memories.json tests/test_harness_flow.py
git commit -m "feat: add mock memory service"
```

### Task 4.3: Integrate Handoff Policy into Harness

**Files:**
- Modify: `app/harness/customer_service_harness.py`
- Test: `tests/test_harness_flow.py`

- [x] **Step 1: Add failing integration test**

Append to `tests/test_harness_flow.py`:

```python
@pytest.mark.asyncio
async def test_harness_applies_handoff_policy_for_empty_reply():
    async def fake_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        return AgentRunResult(final_output="")

    harness = CustomerServiceHarness(executor=BusinessAgentExecutor(runner=fake_runner))
    response = await harness.run(CustomerServiceRequest(tenant_id="tenant_a", content="Help"))

    assert response.need_handoff_to_human is True
    assert response.handoff_type == "reply_handoff"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_harness_applies_handoff_policy_for_empty_reply -v`

Expected: FAIL because harness does not apply policy.

- [x] **Step 3: Modify harness to apply policy**

Update `app/harness/customer_service_harness.py` constructor to include `HandoffPolicy`, then apply it after reply processing:

```python
from app.harness.handoff_policy import HandoffPolicy
```

```python
def __init__(
    self,
    executor: BusinessAgentExecutor | None = None,
    prompt_assembler: PromptAssembler | None = None,
    reply_processor: ReplyPostProcessor | None = None,
    state_reducer: StateReducer | None = None,
    handoff_policy: HandoffPolicy | None = None,
) -> None:
    self.executor = executor or BusinessAgentExecutor()
    self.prompt_assembler = prompt_assembler or PromptAssembler()
    self.reply_processor = reply_processor or ReplyPostProcessor()
    self.state_reducer = state_reducer or StateReducer()
    self.handoff_policy = handoff_policy or HandoffPolicy()
```

After `state.final_reply = ...`, add:

```python
self.handoff_policy.apply(state)
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py tests/test_handoff_policy.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app/harness/customer_service_harness.py tests/test_harness_flow.py
git commit -m "feat: apply handoff policy in harness"
```

### Task 4.4: Add FastAPI Routes with Sync and Async Modes

**Files:**
- Create: `app/main.py`
- Create: `app/api/routes.py`
- Test: `tests/test_demo_cases.py`

- [x] **Step 1: Write failing API tests**

Create `tests/test_demo_cases.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_customer_service_sync_endpoint():
    client = TestClient(app)
    response = client.post(
        "/customer-service/respond",
        json={"tenant_id": "tenant_a", "content": "Hello", "async_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["status"] in {"success", "error"}
```

- [x] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_demo_cases.py -v`

Expected: FAIL with missing `app.main`.

- [x] **Step 3: Implement routes**

Create `app/api/routes.py`:

```python
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks

from app.api.compatibility import convert_emailv4_payload
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.request_models import CustomerServiceRequest
from app.state.result_models import AcceptedResponse, CustomerServiceResponse

router = APIRouter()


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
) -> CustomerServiceResponse | AcceptedResponse:
    if request.async_mode:
        request_id = request.request_id or str(uuid4())
        request.request_id = request_id
        background_tasks.add_task(CustomerServiceHarness().run, request)
        return AcceptedResponse(
            request_id=request_id,
            accepted_at=datetime.now(timezone.utc).isoformat(),
        )
    return await CustomerServiceHarness().run(request)


@router.post("/emailv4")
async def emailv4(payload: dict, background_tasks: BackgroundTasks) -> CustomerServiceResponse | AcceptedResponse:
    request = convert_emailv4_payload(payload)
    return await respond(request, background_tasks)
```

- [x] **Step 4: Implement app entrypoint**

Create `app/main.py`:

```python
from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(router)
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_demo_cases.py -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/main.py app/api/routes.py tests/test_demo_cases.py
git commit -m "feat: add fastapi routes"
```

### Task 4.5: Add Agents Hooks and Optional Langfuse Trace Adapter

**Files:**
- Create: `app/observability/hooks.py`
- Modify: `app/observability/tracing.py`
- Modify: `tests/test_harness_flow.py`

- [x] **Step 1: Add hooks test**

Append to `tests/test_harness_flow.py`:

```python
from app.observability.hooks import CustomerServiceRunHooks


@pytest.mark.asyncio
async def test_customer_service_hooks_record_tool_events():
    state = CustomerServiceState(tenant_id="tenant_a", channel="email", content="Help")
    tracer = LocalTracer(request_id="req_1")
    hooks = CustomerServiceRunHooks(tracer=tracer)

    await hooks.record_tool_event(state, "tool_start", "get_rag_knowledge", {"query": "return"})

    assert tracer.events[0].name == "tool_start"
    assert state.events[0].name == "tool_start"
```

- [x] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_customer_service_hooks_record_tool_events -v`

Expected: FAIL with missing hooks module.

- [x] **Step 3: Add Langfuse optional adapter to tracing**

Extend `app/observability/tracing.py`:

```python
class OptionalLangfuseTracer:
    def __init__(self) -> None:
        self.enabled = False
        self.client = None

    def try_start(self) -> bool:
        try:
            from langfuse import Langfuse

            self.client = Langfuse()
            self.enabled = True
            return True
        except Exception:
            self.enabled = False
            self.client = None
            return False
```

- [x] **Step 4: Implement hook helper**

Create `app/observability/hooks.py`:

```python
from typing import Any

from app.observability.tracing import LocalTracer
from app.state.conversation_state import CustomerServiceState, EventSummary


class CustomerServiceRunHooks:
    def __init__(self, tracer: LocalTracer | None = None) -> None:
        self.tracer = tracer or LocalTracer()

    async def record_tool_event(
        self,
        state: CustomerServiceState,
        event_name: str,
        tool_name: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        payload = {"tool_name": tool_name, **(detail or {})}
        self.tracer.record(event_name, payload)
        state.events.append(EventSummary(name=event_name, detail=payload))
```

- [x] **Step 5: Run hooks test**

Run: `.venv/bin/python -m pytest tests/test_harness_flow.py::test_customer_service_hooks_record_tool_events -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add app/observability/hooks.py app/observability/tracing.py tests/test_harness_flow.py
git commit -m "feat: add customer service observability hooks"
```

### Phase 4 Stop and Report

- [x] Stop after Task 4.5.
- [x] Report changed files, tests run, commits created, and known API/async limitations.
- [x] Wait for user approval before Phase 5.

---

## Phase 5: Demo Cases, Tests, Docs, and Final Verification

**Phase Goal:** Add demo data/scripts, README, full test run, and final polish so the project is easy to run and explain.

### Task 5.1: Add Reply Templates and Demo Case Data

**Files:**
- Create: `data/reply_templates.json`
- Create: `scripts/run_demo_cases.py`

- [ ] **Step 1: Create templates**

Create `data/reply_templates.json`:

```json
{
  "default_email": {
    "subject": "Re: Customer Support",
    "body": "Dear customer,\\n\\nThank you for reaching out. We are reviewing your request and will help as soon as possible.\\n\\nBest regards,\\nSupport Team"
  },
  "handoff_reply": {
    "subject": "Re: Customer Support",
    "body": "Dear customer,\\n\\nThank you for your message. I have escalated this to a human support specialist for further assistance.\\n\\nBest regards,\\nSupport Team"
  }
}
```

- [ ] **Step 2: Create demo script**

Create `scripts/run_demo_cases.py`:

```python
import asyncio
import json

from app.harness.business_agent_executor import AgentRunResult, BusinessAgentExecutor
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.request_models import CustomerProfile, CustomerServiceRequest, SlotDefinition


async def fake_runner(
    state,
    instructions,
    user_message,
    tools,
    mcp_servers,
    max_turns,
    model,
):
    if "human" in state.content.lower():
        state.need_handoff_to_human = True
        state.handoff_type = "reply_handoff"
        state.handoff_reason = "customer requested human support"
    return AgentRunResult(final_output=f"Dear {state.customer.name or 'customer'},\\nWe can help with: {state.content}")


async def main() -> None:
    harness = CustomerServiceHarness(executor=BusinessAgentExecutor(runner=fake_runner))
    cases = [
        CustomerServiceRequest(
            request_id="demo_order",
            tenant_id="tenant_a",
            channel="email",
            subject="Order status",
            content="Where is my order A100?",
            customer=CustomerProfile(name="Ada", id="user_ada"),
            slot_schema=[SlotDefinition(name="order_id", description="Order id")],
        ),
        CustomerServiceRequest(
            request_id="demo_handoff",
            tenant_id="tenant_a",
            channel="chat",
            content="I want to talk to a human agent.",
            customer=CustomerProfile(name="Sam"),
        ),
    ]
    for case in cases:
        response = await harness.run(case)
        print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run demo script**

Run: `.venv/bin/python scripts/run_demo_cases.py`

Expected: prints two JSON responses with `status`.

- [ ] **Step 4: Commit**

```bash
git add data/reply_templates.json scripts/run_demo_cases.py
git commit -m "feat: add demo cases script"
```

### Task 5.2: Add Mock HTTP Server Script

**Files:**
- Create: `scripts/run_mock_http_server.py`

- [ ] **Step 1: Create mock HTTP server**

Create `scripts/run_mock_http_server.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock Customer Service Business APIs")


class OrderRequest(BaseModel):
    order_id: str


@app.post("/mock/order")
async def mock_order(request: OrderRequest) -> dict:
    return {"code": 200, "data": {"order_id": request.order_id, "status": "shipped"}}


@app.post("/mock/logistics")
async def mock_logistics(request: OrderRequest) -> dict:
    return {
        "code": 200,
        "data": {
            "order_id": request.order_id,
            "carrier": "DHL",
            "tracking_number": "DHL123456",
            "status": "in_transit",
        },
    }


@app.get("/mock/refund-policy")
async def mock_refund_policy() -> dict:
    return {"code": 200, "data": {"policy": "Returns are accepted within 30 days."}}
```

- [ ] **Step 2: Syntax check**

Run: `.venv/bin/python -m py_compile scripts/run_mock_http_server.py`

Expected: command exits with code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_mock_http_server.py
git commit -m "feat: add mock business api server"
```

### Task 5.3: Add README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:

```markdown
# OpenAI Agents Intelligent Customer Service

Near-production intelligent customer service Agent backend demo built with `openai-agents-python` and Harness Engineering.

## Capabilities

- FastAPI backend with `/customer-service/respond` and `/emailv4`
- OpenAI Agents SDK execution path
- Structured customer service state
- Config-driven `extract_slots`
- Mock/remote-ready RAG
- Dynamic HTTP tool executor
- MCP manager scaffold and local MCP example server
- Handoff policy
- Sync and async response modes
- Local structured observability with optional Langfuse extension point

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run Demo Cases

```bash
python scripts/run_demo_cases.py
```

## Run Tests

```bash
pytest -v
```

## Design

See `docs/superpowers/specs/2026-06-17-openai-agents-customer-service-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project readme"
```

### Task 5.4: Final Verification

**Files:**
- Modify only if verification finds concrete bugs.

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest -v`

Expected: all tests PASS.

- [ ] **Step 2: Run demo script**

Run: `.venv/bin/python scripts/run_demo_cases.py`

Expected: JSON responses printed for demo cases.

- [ ] **Step 3: Run API import check**

Run: `.venv/bin/python -c "from app.main import app; print(app.title)"`

Expected: prints `OpenAI Agents Customer Service`.

- [ ] **Step 4: Check git status**

Run: `git status --short`

Expected: no unstaged or uncommitted changes.

- [ ] **Step 5: Final phase report**

Report:

- All phase completion status.
- Verification commands and results.
- Commits created.
- Remaining limitations.
- Suggested production hardening next steps.

### Phase 5 Stop and Report

- [ ] Stop after Task 5.4.
- [ ] Report final completion status.
- [ ] Do not merge to `main` unless user explicitly requests it.
