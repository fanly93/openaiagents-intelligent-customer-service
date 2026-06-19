from time import perf_counter
from typing import Any

from app.harness.business_agent_executor import BusinessAgentExecutor
from app.harness.handoff_policy import HandoffPolicy
from app.harness.prompt_assembler import PromptAssembler
from app.harness.reply_post_processor import ReplyPostProcessor
from app.harness.state_reducer import StateReducer
from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer
from app.retrieval.memory_service import MockMemoryService
from app.state.conversation_state import CustomerServiceState, EventSummary
from app.state.request_models import CustomerServiceRequest
from app.state.result_models import CustomerServiceResponse
from app.tools.mcp_manager import MCPManager
from app.tools.registry import ToolRegistry


class CustomerServiceHarness:
    def __init__(
        self,
        executor: BusinessAgentExecutor | None = None,
        prompt_assembler: PromptAssembler | None = None,
        reply_processor: ReplyPostProcessor | None = None,
        state_reducer: StateReducer | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_manager: MCPManager | None = None,
        handoff_policy: HandoffPolicy | None = None,
        memory_service: Any | None = None,
    ) -> None:
        self.executor = executor or BusinessAgentExecutor()
        self.prompt_assembler = prompt_assembler or PromptAssembler()
        self.reply_processor = reply_processor or ReplyPostProcessor()
        self.state_reducer = state_reducer or StateReducer()
        self.tool_registry = tool_registry or ToolRegistry()
        self.mcp_manager = mcp_manager or MCPManager()
        self.handoff_policy = handoff_policy or HandoffPolicy()
        self.memory_service = memory_service or MockMemoryService()

    async def run(
        self,
        request: CustomerServiceRequest,
    ) -> CustomerServiceResponse:
        start = perf_counter()
        tracker = PerformanceTracker()
        tracer = LocalTracer(request_id=request.request_id)
        state = self._init_state(request)

        try:
            tracer.record(
                "request_start",
                {"tenant_id": request.tenant_id, "channel": request.channel},
            )
            with tracker.track("prepare_tools"):
                tool_setup = await self.tool_registry.prepare(request)
            tracer.record(
                "tool_registry_prepare",
                {"enabled_names": tool_setup.enabled_names},
            )
            with tracker.track("prepare_mcp"):
                mcp_setup = await self.mcp_manager.prepare(request.mcp_servers)
            tracer.record(
                "mcp_connect",
                {
                    "configured_names": mcp_setup.enabled_names,
                    "configuration_errors": mcp_setup.errors,
                },
                status="started",
            )
            with tracker.track("memory_retrieve"):
                await self._preload_memory(request, state, tracer)
            with tracker.track("assemble_prompt"):
                guide_parts = [
                    part
                    for part in [tool_setup.tool_guide, mcp_setup.guide_text]
                    if part
                ]
                instructions = self.prompt_assembler.build_instructions(
                    state,
                    tool_guide="\n".join(guide_parts),
                    extra_instructions=request.instructions,
                )
                user_message = self.prompt_assembler.build_user_message(state)

            async with self.mcp_manager.connect(mcp_setup) as active_mcp_servers:
                tracer.record(
                    "mcp_connect",
                    {
                        "active_names": [
                            server.name for server in active_mcp_servers
                        ]
                    },
                )
                tracer.record(
                    "agent_start",
                    {"model": request.model, "max_turns": request.max_turns},
                    status="started",
                )
                with tracker.track("agent_run"):
                    result = await self.executor.run(
                        state=state,
                        instructions=instructions,
                        user_message=user_message,
                        tools=tool_setup.tools,
                        mcp_servers=active_mcp_servers,
                        max_turns=request.max_turns,
                        model=request.model,
                    )
                tracer.record("agent_end", {"token_usage": result.token_usage})

            with tracker.track("post_process"):
                state.final_reply = self.reply_processor.clean(result.final_output)
            with tracker.track("handoff_policy"):
                self.handoff_policy.apply(state)
            tracer.record(
                "handoff_policy",
                {
                    "need_handoff_to_human": state.need_handoff_to_human,
                    "handoff_type": state.handoff_type,
                },
            )
            state.token_usage = result.token_usage
            tracer.record(
                "response_built",
                {"reply_length": len(state.final_reply or "")},
            )
            return self._build_response(state, tracker, tracer, start)
        except Exception as exc:
            self.handoff_policy.apply(state)
            tracer.record(
                "handoff_policy",
                {
                    "need_handoff_to_human": state.need_handoff_to_human,
                    "handoff_type": state.handoff_type,
                    "reason": "agent_execution_failed",
                },
            )
            tracer.record(
                "request_error",
                {"error": str(exc)},
                status="error",
            )
            return self._build_response(
                state,
                tracker,
                tracer,
                start,
                error=str(exc),
            )

    async def _preload_memory(
        self,
        request: CustomerServiceRequest,
        state: CustomerServiceState,
        tracer: LocalTracer,
    ) -> None:
        config = request.memory_config
        customer_id = request.customer.id
        if not config or not config.enabled or not customer_id:
            return

        try:
            state.user_memories = await self.memory_service.retrieve(
                tenant_id=request.tenant_id,
                customer_id=customer_id,
                query=request.content,
                top_k=config.top_k,
            )
        except Exception:
            state.user_memories = []
            tracer.record(
                "memory_retrieve",
                {"customer_id": customer_id, "count": 0},
                status="error",
            )
            return

        tracer.record(
            "memory_retrieve",
            {
                "customer_id": customer_id,
                "count": len(state.user_memories),
            },
        )

    def _build_response(
        self,
        state: CustomerServiceState,
        tracker: PerformanceTracker,
        tracer: LocalTracer,
        start: float,
        error: str | None = None,
    ) -> CustomerServiceResponse:
        state.performance_stats = tracker.summary()
        self._sync_trace_events(state, tracer)
        return self.state_reducer.build_response(
            state,
            processing_time=round(perf_counter() - start, 6),
            error=error,
        )

    @staticmethod
    def _init_state(request: CustomerServiceRequest) -> CustomerServiceState:
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
            EventSummary(
                name=event.name,
                status=event.status,
                detail=event.detail,
            )
            for event in tracer.events
        ]
