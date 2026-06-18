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
        return MCPSetup(
            enabled_names=names,
            guide_text="\n".join(f"- {name}" for name in names),
        )

    def exit_stack(self) -> AsyncExitStack:
        return AsyncExitStack()
