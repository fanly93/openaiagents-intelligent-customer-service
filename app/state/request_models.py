from typing import Any, Literal

from pydantic import BaseModel, Field


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
    timeout: float = Field(default=10.0, ge=0.1, le=30.0)
    max_retries: int = Field(default=0, ge=0, le=3)
    retry_backoff: float = Field(default=0.2, ge=0.0, le=5.0)


class MCPToolParameterOverride(BaseModel):
    name: str
    description: str


class MCPToolOverride(BaseModel):
    name: str
    description: str | None = None
    parameters: list[MCPToolParameterOverride] = Field(default_factory=list)
    parameter_descriptions: dict[str, str] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    name: str
    type: Literal["stdio", "sse", "streamable_http"] = "stdio"
    config: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    tools_filter: dict[str, Any] | None = None
    tools_override: list[MCPToolOverride] = Field(default_factory=list)


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
    max_turns: int = Field(default=6, ge=1, le=20)
    instructions: str | None = None
