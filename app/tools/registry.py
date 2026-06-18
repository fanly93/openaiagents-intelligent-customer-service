import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agents import FunctionTool, RunContextWrapper, function_tool
from agents.tool_context import ToolContext

from app.retrieval.knowledge_retriever import MockKnowledgeRetriever
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import CustomerServiceRequest, HttpToolConfig
from app.tools.core_tools import extract_slots_impl
from app.tools.dynamic_http_tools import call_dynamic_http_tool


@dataclass
class ToolSetup:
    tools: list[Any] = field(default_factory=list)
    tool_guide: str = ""
    enabled_names: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(
        self,
        knowledge_retriever: MockKnowledgeRetriever | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.knowledge_retriever = knowledge_retriever or MockKnowledgeRetriever()
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).resolve().parents[2] / "data"
        )

    async def prepare(self, request: CustomerServiceRequest) -> ToolSetup:
        tools = [
            self._extract_slots_tool(),
            self._rag_tool(request),
            self._handoff_tool(),
        ]

        optional_factories = {
            "check_order_status": self._order_tool,
            "check_shipping_status": self._shipping_tool,
        }
        for name in request.tools:
            factory = optional_factories.get(name)
            if factory and name not in {tool.name for tool in tools}:
                tools.append(factory())

        for config in request.http_tools:
            if config.name not in {tool.name for tool in tools}:
                tools.append(self._dynamic_http_tool(config))

        enabled_names = [tool.name for tool in tools]
        guide = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        )
        return ToolSetup(
            tools=tools,
            tool_guide=guide,
            enabled_names=enabled_names,
        )

    @staticmethod
    def _extract_slots_tool() -> FunctionTool:
        async def extract_slots(
            context: RunContextWrapper[CustomerServiceState],
            slots: list[dict[str, Any]],
        ) -> dict[str, list[str]]:
            """Validate and store configured customer-service slots."""
            return await extract_slots_impl(context.context, slots)

        return function_tool(
            extract_slots,
            name_override="extract_slots",
            strict_mode=False,
        )

    def _rag_tool(self, request: CustomerServiceRequest) -> FunctionTool:
        async def get_rag_knowledge(
            context: RunContextWrapper[CustomerServiceState],
            query: str,
        ) -> list[dict[str, Any]]:
            """Retrieve tenant-scoped knowledge relevant to the query."""
            try:
                result = await self.knowledge_retriever.retrieve(
                    tenant_id=context.context.tenant_id,
                    query=query,
                    knowledge_config=request.knowledge_config,
                )
            except Exception:
                result = []
                context.context.retrieved_knowledge = result
                context.context.record_tool_call(
                    "get_rag_knowledge",
                    {"query": query},
                    {"error": "rag_retrieval_failed"},
                )
                return result
            context.context.retrieved_knowledge = result
            context.context.record_tool_call(
                "get_rag_knowledge",
                {"query": query},
                result,
            )
            return result

        return function_tool(
            get_rag_knowledge,
            name_override="get_rag_knowledge",
        )

    @staticmethod
    def _handoff_tool() -> FunctionTool:
        async def handoff_to_human(
            context: RunContextWrapper[CustomerServiceState],
            handoff_type: Literal["reply_handoff", "no_reply_handoff"],
            reason: str,
        ) -> dict[str, Any]:
            """Escalate the current request to a human support agent."""
            state = context.context
            state.need_handoff_to_human = True
            state.handoff_type = handoff_type
            state.handoff_reason = reason
            result = {
                "need_handoff_to_human": True,
                "handoff_type": handoff_type,
                "reason": reason,
            }
            state.record_tool_call("handoff_to_human", result, result)
            return result

        return function_tool(
            handoff_to_human,
            name_override="handoff_to_human",
        )

    def _order_tool(self) -> FunctionTool:
        orders = self._load_json("mock_orders.json")

        async def check_order_status(
            context: RunContextWrapper[CustomerServiceState],
            order_id: str,
        ) -> dict[str, Any]:
            """Look up an order by order id."""
            result = orders.get(
                order_id,
                {"error": "order_not_found", "order_id": order_id},
            )
            context.context.order_result = result
            context.context.record_tool_call(
                "check_order_status",
                {"order_id": order_id},
                result,
            )
            return result

        return function_tool(
            check_order_status,
            name_override="check_order_status",
        )

    def _shipping_tool(self) -> FunctionTool:
        logistics = self._load_json("mock_logistics.json")

        async def check_shipping_status(
            context: RunContextWrapper[CustomerServiceState],
            order_id: str,
        ) -> dict[str, Any]:
            """Look up shipment status by order id."""
            result = logistics.get(
                order_id,
                {"error": "shipment_not_found", "order_id": order_id},
            )
            context.context.logistics_result = result
            context.context.record_tool_call(
                "check_shipping_status",
                {"order_id": order_id},
                result,
            )
            return result

        return function_tool(
            check_shipping_status,
            name_override="check_shipping_status",
        )

    @staticmethod
    def _dynamic_http_tool(config: HttpToolConfig) -> FunctionTool:
        json_types = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
        }
        properties = {
            param.name: {
                "type": json_types[param.type],
                "description": param.description,
            }
            for param in config.request_params
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": [
                param.name for param in config.request_params if param.required
            ],
            "additionalProperties": not bool(config.request_params),
        }

        async def invoke(context: ToolContext[Any], raw_input: str) -> str:
            try:
                params = json.loads(raw_input or "{}")
            except json.JSONDecodeError:
                params = None
            if not isinstance(params, dict):
                result = {"error": "invalid_tool_input"}
                context.context.record_tool_call(
                    config.name,
                    {
                        "input_type": "invalid_json",
                        "input_length": len(raw_input),
                    },
                    result,
                )
                return json.dumps(
                    result,
                    ensure_ascii=False,
                )
            result = await call_dynamic_http_tool(
                context.context,
                config,
                params,
            )
            return json.dumps(result, ensure_ascii=False)

        return FunctionTool(
            name=config.name,
            description=config.description,
            params_json_schema=schema,
            on_invoke_tool=invoke,
            strict_json_schema=False,
        )

    def _load_json(self, filename: str) -> dict[str, Any]:
        path = self.data_dir / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
