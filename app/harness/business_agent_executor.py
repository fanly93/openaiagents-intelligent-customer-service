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
        hooks: Any | None = None,
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
            hooks,
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
        hooks: Any | None,
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
            hooks=hooks,
            max_turns=max_turns,
        )

        aggregate_usage = getattr(
            getattr(result, "context_wrapper", None),
            "usage",
            None,
        )
        token_usage = (
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
            token_usage=token_usage,
            raw_responses=getattr(result, "raw_responses", []) or [],
        )
