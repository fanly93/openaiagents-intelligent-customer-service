import json
import sys
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from app.harness.business_agent_executor import AgentRunResult, BusinessAgentExecutor
from app.harness.customer_service_harness import CustomerServiceHarness
from app.harness.prompt_assembler import PromptAssembler
from app.harness.state_reducer import StateReducer
from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import (
    CustomerProfile,
    CustomerServiceRequest,
    HttpToolConfig,
    HttpToolParam,
    MCPServerConfig,
    MCPToolOverride,
    SlotDefinition,
)
from app.tools.mcp_manager import MCPManager, MCPServerDefinition
from app.tools.registry import ToolRegistry


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


def test_prompt_assembler_includes_runtime_context_and_language_rules():
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


def test_state_reducer_redacts_sensitive_snapshot_values():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="Help",
        http_tool_results={
            "lookup": {
                "access_token": "secret-value",
                "profile": {"email": "buyer@example.com"},
            }
        },
    )

    response = StateReducer().build_response(state, processing_time=0.1)

    result = response.state_snapshot["http_tool_results"]["lookup"]
    assert result["access_token"] == "[REDACTED]"
    assert result["profile"]["email"] == "[REDACTED]"
    assert state.http_tool_results["lookup"]["access_token"] == "secret-value"


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
        return AgentRunResult(
            final_output="Hello from mock",
            token_usage={"total_tokens": 10},
        )

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


@pytest.mark.asyncio
async def test_business_agent_executor_uses_sdk_model_and_aggregate_usage(monkeypatch):
    import agents

    state = CustomerServiceState(tenant_id="tenant_a", channel="chat", content="Hi")

    async def fake_sdk_run(starting_agent, input, **kwargs):
        assert starting_agent.model == "gpt-test"
        assert input == "Hi"
        assert kwargs["context"] is state
        assert kwargs["max_turns"] == 3
        return SimpleNamespace(
            final_output="Hello from SDK",
            raw_responses=["response"],
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=7,
                    output_tokens=5,
                    total_tokens=12,
                )
            ),
        )

    monkeypatch.setattr(agents.Runner, "run", fake_sdk_run)

    result = await BusinessAgentExecutor().run(
        state=state,
        instructions="instructions",
        user_message="Hi",
        tools=[],
        mcp_servers=[],
        max_turns=3,
        model="gpt-test",
    )

    assert result.final_output == "Hello from SDK"
    assert result.token_usage == {
        "input_tokens": 7,
        "output_tokens": 5,
        "total_tokens": 12,
    }


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
        return AgentRunResult(
            final_output=(
                "THINK: internal reasoning\n"
                "Dear Ada,\n"
                "Your order has shipped."
            ),
            token_usage={"total_tokens": 10},
        )

    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=fake_runner)
    )
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
    assert "THINK:" not in response.reply["body"]
    assert response.token_usage["total_tokens"] == 10
    assert response.state_snapshot["events"][0]["name"] == "request_start"
    assert "agent_run" in response.state_snapshot["performance_stats"]

@pytest.mark.asyncio
async def test_customer_service_harness_returns_traced_error_response():
    async def failing_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        raise RuntimeError("model unavailable")

    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=failing_runner)
    )
    response = await harness.run(
        CustomerServiceRequest(
            request_id="req_error",
            tenant_id="tenant_a",
            content="Help",
        )
    )

    assert response.status == "error"
    assert response.error == "model unavailable"
    assert response.state_snapshot["events"][-1]["name"] == "request_error"
    assert response.state_snapshot["events"][-1]["status"] == "error"
    assert "agent_run" in response.state_snapshot["performance_stats"]


@pytest.mark.asyncio
async def test_tool_registry_builds_real_core_and_request_scoped_tools():
    setup = await ToolRegistry().prepare(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Where is order A100?",
            tools=["check_order_status"],
            http_tools=[
                HttpToolConfig(
                    name="lookup_return_case",
                    description="Look up a return case",
                    url="https://mock.local/returns",
                    request_params=[
                        HttpToolParam(
                            name="case_id",
                            description="Return case id",
                            required=True,
                        )
                    ],
                )
            ],
        )
    )

    assert setup.enabled_names == [
        "extract_slots",
        "get_rag_knowledge",
        "handoff_to_human",
        "check_order_status",
        "lookup_return_case",
    ]
    assert [tool.name for tool in setup.tools] == setup.enabled_names
    assert setup.tools[-1].params_json_schema["required"] == ["case_id"]
    assert "lookup_return_case" in setup.tool_guide


@pytest.mark.asyncio
async def test_order_tool_reads_mock_data_and_updates_state():
    setup = await ToolRegistry().prepare(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Where is order A100?",
            tools=["check_order_status"],
        )
    )
    order_tool = next(tool for tool in setup.tools if tool.name == "check_order_status")
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Where is order A100?",
    )

    result = await order_tool.on_invoke_tool(
        ToolContext(
            context=state,
            tool_name=order_tool.name,
            tool_call_id="call_1",
            tool_arguments='{"order_id":"A100"}',
        ),
        json.dumps({"order_id": "A100"}),
    )

    assert result["status"] == "shipped"
    assert state.order_result["order_id"] == "A100"
    assert state.tool_call_history[-1]["tool_name"] == "check_order_status"


@pytest.mark.asyncio
async def test_harness_passes_prepared_tools_and_guide_to_executor():
    async def fake_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        assert [tool.name for tool in tools][:3] == [
            "extract_slots",
            "get_rag_knowledge",
            "handoff_to_human",
        ]
        assert "check_order_status" in [tool.name for tool in tools]
        assert "check_order_status" in instructions
        return AgentRunResult(final_output="Order checked.")

    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=fake_runner)
    )
    response = await harness.run(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Where is order A100?",
            tools=["check_order_status"],
        )
    )

    assert response.status == "success"


@pytest.mark.asyncio
async def test_mcp_manager_builds_supported_transports_and_filters():
    manager = MCPManager(
        server_registry={
            "product_support": MCPServerDefinition(
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": ["mcp_servers/product_support_server.py"],
                },
            ),
            "remote_sse": MCPServerDefinition(
                type="sse",
                config={"url": "https://mcp.example.com/sse"},
            ),
            "remote_http": MCPServerDefinition(
                type="streamable_http",
                config={"url": "https://mcp.example.com/mcp"},
            ),
        }
    )
    setup = await manager.prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="stdio",
                config={
                    "command": "/bin/sh",
                    "args": ["-c", "echo compromised"],
                },
                tools_filter={
                    "allowed_tool_names": ["lookup_product_manual"],
                    "blocked_tool_names": ["check_warranty_policy"],
                },
            ),
            MCPServerConfig(
                name="remote_sse",
                type="sse",
                config={"url": "https://mcp.example.com/sse"},
                headers={"Authorization": "Bearer token"},
            ),
            MCPServerConfig(
                name="remote_http",
                type="streamable_http",
                config={"url": "https://mcp.example.com/mcp"},
            ),
        ]
    )

    assert setup.enabled_names == [
        "product_support",
        "remote_sse",
        "remote_http",
    ]
    assert len(setup.servers) == 3
    assert setup.servers[0].params.command == sys.executable
    assert setup.servers[0].params.args == [
        "mcp_servers/product_support_server.py"
    ]
    assert setup.servers[0].tool_filter == {
        "allowed_tool_names": ["lookup_product_manual"],
        "blocked_tool_names": ["check_warranty_policy"],
    }


@pytest.mark.asyncio
async def test_local_stdio_mcp_server_is_runnable():
    manager = MCPManager()
    setup = await manager.prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": ["mcp_servers/product_support_server.py"],
                },
            )
        ]
    )

    async with manager.connect(setup) as active_servers:
        assert len(active_servers) == 1
        tools = await active_servers[0].list_tools()

    assert {tool.name for tool in tools} == {
        "lookup_product_manual",
        "check_warranty_policy",
    }


@pytest.mark.asyncio
async def test_local_stdio_mcp_server_applies_tool_filter():
    manager = MCPManager()
    setup = await manager.prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": ["mcp_servers/product_support_server.py"],
                },
                tools_filter={
                    "allowed_tool_names": ["lookup_product_manual"],
                },
            )
        ]
    )

    async with manager.connect(setup) as active_servers:
        tools = await active_servers[0].list_tools()

    assert [tool.name for tool in tools] == ["lookup_product_manual"]


@pytest.mark.asyncio
async def test_local_stdio_mcp_server_applies_tool_overrides():
    manager = MCPManager()
    setup = await manager.prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": ["mcp_servers/product_support_server.py"],
                },
                tools_override=[
                    {
                        "name": "lookup_product_manual",
                        "description": "Tenant-specific manual lookup.",
                        "parameters": [
                            {
                                "name": "question",
                                "description": "Customer's exact support question.",
                            }
                        ],
                    }
                ],
            )
        ]
    )

    async with manager.connect(setup) as active_servers:
        tools = await active_servers[0].list_tools()

    tool = next(item for item in tools if item.name == "lookup_product_manual")
    assert tool.description == "Tenant-specific manual lookup."
    assert (
        tool.inputSchema["properties"]["question"]["description"]
        == "Customer's exact support question."
    )


@pytest.mark.asyncio
async def test_mcp_manager_skips_invalid_config_without_dropping_valid_servers():
    setup = await MCPManager().prepare(
        [
            MCPServerConfig(
                name="invalid",
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": None,
                    "timeout": "not-a-number",
                },
            ),
            MCPServerConfig(
                name="valid",
                type="stdio",
                config={
                    "command": sys.executable,
                    "args": ["mcp_servers/product_support_server.py"],
                },
            ),
        ]
    )

    assert setup.enabled_names == []
    assert {item["name"] for item in setup.errors} == {"invalid", "valid"}


def test_mcp_tool_override_rejects_malformed_parameter_shapes():
    with pytest.raises(ValueError):
        MCPToolOverride.model_validate(
            {
                "name": "lookup_product_manual",
                "parameters": {"question": "not-a-list"},
            }
        )


@pytest.mark.asyncio
async def test_mcp_manager_reports_type_mismatch_without_building_server():
    setup = await MCPManager().prepare(
        [
            MCPServerConfig(
                name="product_support",
                type="sse",
            )
        ]
    )

    assert setup.servers == []
    assert setup.errors == [
        {
            "name": "product_support",
            "error": "server_type_mismatch",
        }
    ]


@pytest.mark.asyncio
async def test_mcp_manager_reports_missing_trusted_server_endpoint():
    manager = MCPManager(
        server_registry={
            "broken_remote": MCPServerDefinition(
                type="streamable_http",
                config={},
            )
        }
    )

    setup = await manager.prepare(
        [
            MCPServerConfig(
                name="broken_remote",
                type="streamable_http",
            )
        ]
    )

    assert setup.servers == []
    assert setup.errors == [
        {
            "name": "broken_remote",
            "error": "trusted MCP definition requires a url",
        }
    ]


@pytest.mark.asyncio
async def test_harness_passes_connected_mcp_servers_to_executor():
    async def fake_runner(
        state,
        instructions,
        user_message,
        tools,
        mcp_servers,
        max_turns,
        model,
    ):
        assert len(mcp_servers) == 1
        assert mcp_servers[0].name == "product_support"
        assert "product_support" in instructions
        return AgentRunResult(final_output="Manual checked.")

    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=fake_runner)
    )
    response = await harness.run(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="How do I fix E01?",
            mcp_servers=[
                MCPServerConfig(
                    name="product_support",
                    type="stdio",
                    config={
                        "command": sys.executable,
                        "args": ["mcp_servers/product_support_server.py"],
                    },
                )
            ],
        )
    )

    assert response.status == "success"


@pytest.mark.asyncio
async def test_rag_tool_degrades_to_empty_result_on_provider_failure():
    class FailingRetriever:
        async def retrieve(self, **kwargs):
            raise RuntimeError("rag unavailable")

    setup = await ToolRegistry(
        knowledge_retriever=FailingRetriever()
    ).prepare(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Return policy?",
        )
    )
    rag_tool = next(
        tool for tool in setup.tools if tool.name == "get_rag_knowledge"
    )
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Return policy?",
    )

    result = await rag_tool.on_invoke_tool(
        ToolContext(
            context=state,
            tool_name=rag_tool.name,
            tool_call_id="rag_1",
            tool_arguments='{"query":"return policy"}',
        ),
        '{"query":"return policy"}',
    )

    assert result == []
    assert state.retrieved_knowledge == []
    assert state.tool_call_history[-1]["result"]["error"] == "rag_retrieval_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_input",
    ["not-json", "[]", "null"],
)
async def test_dynamic_function_tool_returns_structured_input_error(raw_input):
    setup = await ToolRegistry().prepare(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Look up return",
            http_tools=[
                HttpToolConfig(
                    name="lookup_return",
                    description="Look up return",
                    url="https://mock.local/returns",
                )
            ],
        )
    )
    tool = next(item for item in setup.tools if item.name == "lookup_return")
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Look up return",
    )

    raw_result = await tool.on_invoke_tool(
        ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="http_1",
            tool_arguments=raw_input,
        ),
        raw_input,
    )

    assert json.loads(raw_result)["error"] == "invalid_tool_input"
    assert state.tool_call_history[-1]["tool_name"] == "lookup_return"
    assert state.tool_call_history[-1]["result"] == {
        "error": "invalid_tool_input"
    }
