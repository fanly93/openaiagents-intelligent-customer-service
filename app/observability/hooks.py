from typing import Any

from agents import RunHooks

from app.observability.tracing import LocalTracer
from app.state.conversation_state import CustomerServiceState, EventSummary


class CustomerServiceRunHooks(RunHooks[CustomerServiceState]):
    def __init__(self, tracer: LocalTracer | None = None) -> None:
        self.tracer = tracer or LocalTracer()

    async def record_tool_event(
        self,
        state: CustomerServiceState,
        event_name: str,
        tool_name: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            state,
            event_name,
            {"tool_name": tool_name, **(detail or {})},
        )

    async def on_agent_start(
        self,
        context: Any,
        agent: Any,
    ) -> None:
        self._record(
            context.context,
            "agent_start",
            {"agent_name": getattr(agent, "name", "unknown")},
            status="started",
        )

    async def on_agent_end(
        self,
        context: Any,
        agent: Any,
        output: Any,
    ) -> None:
        self._record(
            context.context,
            "agent_end",
            {
                "agent_name": getattr(agent, "name", "unknown"),
                "output_length": len(str(output)),
            },
        )

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Any,
    ) -> None:
        await self.record_tool_event(
            context.context,
            "tool_start",
            getattr(tool, "name", "unknown"),
        )

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: object,
    ) -> None:
        await self.record_tool_event(
            context.context,
            "tool_end",
            getattr(tool, "name", "unknown"),
            {"result_type": type(result).__name__},
        )

    async def on_handoff(
        self,
        context: Any,
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        self._record(
            context.context,
            "handoff",
            {
                "from_agent": getattr(from_agent, "name", "unknown"),
                "to_agent": getattr(to_agent, "name", "unknown"),
            },
        )

    def _record(
        self,
        state: CustomerServiceState,
        name: str,
        detail: dict[str, Any],
        status: str = "success",
    ) -> None:
        self.tracer.record(name, detail, status=status)
        state.events.append(
            EventSummary(
                name=name,
                status=status,
                detail=detail,
            )
        )
