from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

from agents.mcp import (
    MCPServerManager,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)

from app.state.request_models import MCPServerConfig, MCPToolOverride


@dataclass(frozen=True)
class MCPServerDefinition:
    type: str
    config: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    allow_request_tool_overrides: bool = False


@dataclass
class MCPSetup:
    servers: list[Any] = field(default_factory=list)
    enabled_names: list[str] = field(default_factory=list)
    guide_text: str = ""
    tool_overrides: dict[str, list[MCPToolOverride]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


class MCPServerOverrideProxy:
    def __init__(
        self,
        server: Any,
        overrides: list[MCPToolOverride],
    ) -> None:
        self._server = server
        self._overrides = {
            item.name: item
            for item in overrides
        }

    @property
    def name(self) -> str:
        return self._server.name

    async def connect(self) -> Any:
        return await self._server.connect()

    async def cleanup(self) -> Any:
        return await self._server.cleanup()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        return await self._server.call_tool(tool_name, arguments, meta)

    async def list_tools(
        self,
        run_context: Any = None,
        agent: Any = None,
    ) -> list[Any]:
        tools = await self._server.list_tools(run_context, agent)
        return [self._apply_override(tool) for tool in tools]

    def _apply_override(self, tool: Any) -> Any:
        override = self._overrides.get(tool.name)
        if not override:
            return tool

        input_schema = deepcopy(tool.inputSchema)
        properties = input_schema.get("properties", {})
        for parameter in override.parameters:
            name = parameter.name
            description = parameter.description
            if name in properties and description:
                properties[name]["description"] = str(description)

        for name, description in override.parameter_descriptions.items():
            if name in properties and description:
                properties[name]["description"] = str(description)

        return tool.model_copy(
            update={
                "description": override.description or tool.description,
                "inputSchema": input_schema,
            }
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server, name)


class MCPManager:
    def __init__(
        self,
        server_registry: (
            dict[str, dict[str, MCPServerDefinition]] | None
        ) = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_registry = {
            "tenant_a": {
                "product_support": MCPServerDefinition(
                    type="stdio",
                    config={
                        "command": sys.executable,
                        "args": [
                            str(
                                project_root
                                / "mcp_servers/product_support_server.py"
                            )
                        ],
                    },
                )
            }
        }
        source_registry = (
            default_registry if server_registry is None else server_registry
        )
        self.server_registry = deepcopy(source_registry)

    async def prepare(
        self,
        tenant_id: str,
        configs: list[MCPServerConfig],
    ) -> MCPSetup:
        servers: list[Any] = []
        enabled_names: list[str] = []
        guide_lines: list[str] = []
        tool_overrides: dict[str, list[MCPToolOverride]] = {}
        errors: list[dict[str, str]] = []
        tenant_registry = self.server_registry.get(tenant_id, {})

        for config in configs:
            definition = tenant_registry.get(config.name)
            if definition is None:
                errors.append(
                    {"name": config.name, "error": "untrusted_server"}
                )
                continue
            if definition.type != config.type:
                errors.append(
                    {"name": config.name, "error": "server_type_mismatch"}
                )
                continue
            try:
                server, endpoint = self._build_server(config, definition)
            except (TypeError, ValueError) as exc:
                errors.append(
                    {"name": config.name, "error": str(exc)}
                )
                continue
            if server is None:
                continue
            if (
                config.tools_override
                and definition.allow_request_tool_overrides
            ):
                server = MCPServerOverrideProxy(server, config.tools_override)
                tool_overrides[config.name] = config.tools_override
            elif config.tools_override:
                errors.append(
                    {
                        "name": config.name,
                        "error": "tool_overrides_not_allowed",
                    }
                )

            servers.append(server)
            enabled_names.append(config.name)
            guide_lines.append(f"- {config.name} ({config.type}): {endpoint}")

        return MCPSetup(
            servers=servers,
            enabled_names=enabled_names,
            guide_text="\n".join(guide_lines),
            tool_overrides=tool_overrides,
            errors=errors,
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

    @classmethod
    def _tool_filter(
        cls,
        trusted_config: dict[str, Any] | None,
        request_config: dict[str, Any] | None,
    ) -> Any:
        merged = cls._restrict_tool_filter(
            trusted_config,
            request_config,
        )
        if not merged:
            return None
        return create_static_tool_filter(
            allowed_tool_names=merged.get("allowed_tool_names"),
            blocked_tool_names=merged.get("blocked_tool_names"),
        )

    @staticmethod
    def _filter_names(
        config: dict[str, Any] | None,
        keys: tuple[str, ...],
    ) -> list[str] | None:
        if not config:
            return None
        for key in keys:
            value = config.get(key)
            if value is not None:
                if not isinstance(value, list):
                    raise TypeError("MCP tool filter must use lists")
                return [str(item) for item in value]
        return None

    @classmethod
    def _restrict_tool_filter(
        cls,
        trusted_config: dict[str, Any] | None,
        request_config: dict[str, Any] | None,
    ) -> dict[str, list[str] | None] | None:
        allowed_keys = (
            "allowed_tool_names",
            "allowlist",
            "whitelist",
        )
        blocked_keys = (
            "blocked_tool_names",
            "blocklist",
            "blacklist",
        )
        trusted_allowed = cls._filter_names(
            trusted_config,
            allowed_keys,
        )
        request_allowed = cls._filter_names(
            request_config,
            allowed_keys,
        )
        trusted_blocked = cls._filter_names(
            trusted_config,
            blocked_keys,
        )
        request_blocked = cls._filter_names(
            request_config,
            blocked_keys,
        )

        if trusted_allowed is None:
            allowed = request_allowed
        elif request_allowed is None:
            allowed = trusted_allowed
        else:
            requested = set(request_allowed)
            allowed = [
                name for name in trusted_allowed if name in requested
            ]

        blocked = list(
            dict.fromkeys(
                (trusted_blocked or []) + (request_blocked or [])
            )
        )
        if allowed is None and not blocked:
            return None
        return {
            "allowed_tool_names": allowed,
            "blocked_tool_names": blocked or None,
        }

    def _build_server(
        self,
        config: MCPServerConfig,
        definition: MCPServerDefinition,
    ) -> tuple[Any | None, str]:
        tool_filter = self._tool_filter(
            definition.config.get("tools_filter"),
            config.tools_filter,
        )
        timeout = float(definition.config.get("timeout", 30))
        if not 0.1 <= timeout <= 60:
            raise ValueError("MCP timeout must be between 0.1 and 60 seconds")

        if definition.type == "stdio":
            command = definition.config.get("command")
            if not command:
                raise ValueError(
                    "trusted MCP definition requires a command"
                )
            raw_args = definition.config.get("args", [])
            raw_env = definition.config.get("env", {})
            if not isinstance(raw_args, list) or not isinstance(raw_env, dict):
                raise TypeError("stdio args/env have invalid types")
            return (
                MCPServerStdio(
                    params={
                        "command": str(command),
                        "args": [str(item) for item in raw_args],
                        "env": {
                            str(key): str(value)
                            for key, value in raw_env.items()
                        },
                    },
                    name=config.name,
                    client_session_timeout_seconds=timeout,
                    tool_filter=tool_filter,
                    cache_tools_list=True,
                ),
                str(command),
            )

        url = definition.config.get("url")
        if not url:
            raise ValueError("trusted MCP definition requires a url")
        params = {
            "url": str(url),
            "headers": dict(definition.headers),
            "timeout": timeout,
        }
        server_class = (
            MCPServerSse
            if definition.type == "sse"
            else MCPServerStreamableHttp
        )
        return (
            server_class(
                params=params,
                name=config.name,
                client_session_timeout_seconds=timeout,
                tool_filter=tool_filter,
                cache_tools_list=True,
            ),
            str(url),
        )
