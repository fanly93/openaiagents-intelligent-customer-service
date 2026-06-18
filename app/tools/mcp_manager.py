from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agents.mcp import (
    MCPServerManager,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)

from app.state.request_models import MCPServerConfig


@dataclass
class MCPSetup:
    servers: list[Any] = field(default_factory=list)
    enabled_names: list[str] = field(default_factory=list)
    guide_text: str = ""
    tool_overrides: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class MCPManager:
    async def prepare(self, configs: list[MCPServerConfig]) -> MCPSetup:
        servers: list[Any] = []
        enabled_names: list[str] = []
        guide_lines: list[str] = []
        tool_overrides: dict[str, list[dict[str, Any]]] = {}

        for config in configs:
            tool_filter = self._tool_filter(config.tools_filter)
            timeout = float(config.config.get("timeout", 30))

            if config.type == "stdio":
                command = config.config.get("command")
                if not command:
                    continue
                server = MCPServerStdio(
                    params={
                        "command": str(command),
                        "args": [
                            str(item) for item in config.config.get("args", [])
                        ],
                        "env": {
                            str(key): str(value)
                            for key, value in config.config.get("env", {}).items()
                        },
                    },
                    name=config.name,
                    client_session_timeout_seconds=timeout,
                    tool_filter=tool_filter,
                    cache_tools_list=True,
                )
                endpoint = str(command)
            elif config.type == "sse":
                url = config.config.get("url")
                if not url:
                    continue
                server = MCPServerSse(
                    params={
                        "url": str(url),
                        "headers": config.headers,
                        "timeout": timeout,
                    },
                    name=config.name,
                    client_session_timeout_seconds=timeout,
                    tool_filter=tool_filter,
                    cache_tools_list=True,
                )
                endpoint = str(url)
            else:
                url = config.config.get("url")
                if not url:
                    continue
                server = MCPServerStreamableHttp(
                    params={
                        "url": str(url),
                        "headers": config.headers,
                        "timeout": timeout,
                    },
                    name=config.name,
                    client_session_timeout_seconds=timeout,
                    tool_filter=tool_filter,
                    cache_tools_list=True,
                )
                endpoint = str(url)

            servers.append(server)
            enabled_names.append(config.name)
            guide_lines.append(f"- {config.name} ({config.type}): {endpoint}")
            if config.tools_override:
                tool_overrides[config.name] = config.tools_override

        return MCPSetup(
            servers=servers,
            enabled_names=enabled_names,
            guide_text="\n".join(guide_lines),
            tool_overrides=tool_overrides,
        )

    @asynccontextmanager
    async def connect(self, setup: MCPSetup) -> AsyncIterator[list[Any]]:
        if not setup.servers:
            yield []
            return

        async with MCPServerManager(
            setup.servers,
            drop_failed_servers=True,
            strict=False,
            connect_in_parallel=True,
        ) as manager:
            yield list(manager.active_servers)

    @staticmethod
    def _tool_filter(config: dict[str, Any] | None) -> Any:
        if not config:
            return None
        allowed = (
            config.get("allowed_tool_names")
            or config.get("allowlist")
            or config.get("whitelist")
        )
        blocked = (
            config.get("blocked_tool_names")
            or config.get("blocklist")
            or config.get("blacklist")
        )
        return create_static_tool_filter(
            allowed_tool_names=allowed,
            blocked_tool_names=blocked,
        )
