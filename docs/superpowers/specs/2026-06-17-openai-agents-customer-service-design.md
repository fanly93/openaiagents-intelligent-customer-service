# OpenAI Agents 智能客服后端设计

日期：2026-06-17

项目目录：

`/Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service`

参考源项目主流程：

`/Users/tanglin/VibeCoding/intelligent-customer-service/lmas-email-agent/core/single_react_agent_1118.py`

源项目探索记录：

`/Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service/findings/single_react_agent_1118_source_findings.md`

## 1. 项目目标

本项目要实现一个接近生产型的智能客服 Agent 后端 Demo。系统面向中国品牌出海业务，主要覆盖邮件客服，同时预留在线客服 chat 扩展能力。第一版要真实使用 `openai-agents-python` 执行 Agent 主流程，并展示工具调用、RAG、MCP、动态 HTTP 工具、转人工、异步 webhook 和链路观测。

第一版不是完整生产系统，但代码边界必须按生产演进设计。Mock 数据只作为 provider 实现，不应污染核心 harness。后续可以无缝替换为真实订单系统、物流系统、RAG 服务、记忆服务、模板数据库、工单系统和观测平台。

## 2. 第一版范围

第一版实现：

- 基于 OpenAI Agents SDK 的 `Agent`、`Runner.run()`、tools、hooks 构建主流程。
- 使用 Harness Engineering 拆分 Agent 生命周期。
- 支持 `email` 和 `chat` 渠道，邮件为主场景。
- 提供新接口 `/customer-service/respond` 和兼容接口 `/emailv4`。
- 默认使用本地 mock 订单、物流、RAG、记忆和模板数据。
- 动态 HTTP 工具使用真实 HTTP 能力，测试时通过本地 mock server 演示。
- MCP 内置一个本地 `stdio` 示例 server，并支持配置接入外部 MCP server。
- RAG 默认本地 mock 检索，同时预留远程 RAG 适配接口。
- Langfuse 环境变量存在时上报 trace/span，不存在时降级为本地结构化事件。
- 准备 8-12 个演示场景脚本。

第一版不实现：

- 前端 UI。
- 真实企业生产订单/物流系统接入。
- 完整评测平台。
- 自动 fileIds 路由算法。
- 完整 SaaS 多租户权限、计费和管理后台。

## 3. 建议目录结构

```text
openaiagents-intelligent-customer-service/
  app/
    api/
      routes.py
      compatibility.py
    harness/
      customer_service_harness.py
      business_agent_executor.py
      prompt_assembler.py
      reply_post_processor.py
      state_reducer.py
      handoff_policy.py
    state/
      request_models.py
      conversation_state.py
      result_models.py
    tools/
      registry.py
      core_tools.py
      dynamic_http_tools.py
      mcp_manager.py
    retrieval/
      knowledge_retriever.py
      memory_service.py
    observability/
      hooks.py
      tracing.py
      perf.py
    config.py
    main.py
  mcp_servers/
    product_support_server.py
  data/
    knowledge/
    mock_orders.json
    mock_logistics.json
    mock_memories.json
    reply_templates.json
  scripts/
    run_demo_cases.py
    run_mock_http_server.py
  tests/
  findings/
  docs/
  pyproject.toml
  README.md
```

模块职责：

- `api/`：FastAPI 路由、旧接口兼容、异步 webhook，不写 Agent 业务逻辑。
- `harness/`：Agent 生命周期编排、状态归并、转人工、后处理。
- `state/`：请求模型、上下文状态、响应模型。
- `tools/`：核心工具、动态 HTTP 工具、MCP 管理。
- `retrieval/`：企业知识库 RAG 和用户长期记忆。
- `observability/`：本地事件、Langfuse、hooks、耗时统计。
- `data/`：本地 mock 数据和模板。
- `scripts/`：演示脚本和本地 mock HTTP server。

## 4. 请求、状态和响应模型

### 4.1 CustomerServiceRequest

主接口 `/customer-service/respond` 使用统一请求模型：

```python
CustomerServiceRequest:
  request_id: str | None
  tenant_id: str
  channel: Literal["email", "chat"]
  subject: str | None
  content: str
  customer: CustomerProfile
  contexts: list[ConversationTurn]
  tools: list[str]
  slot_schema: list[SlotDefinition]
  http_tools: list[HttpToolConfig]
  mcp_servers: list[MCPServerConfig]
  knowledge_config: KnowledgeConfig | None
  memory_config: MemoryConfig | None
  webhook_url: str | None
  async_mode: bool
  model: str | None
  max_turns: int
  instructions: str | None
```

兼容接口 `/emailv4` 将旧字段映射为新模型：

- `corp` -> `tenant_id`
- `content` -> `content`
- `customer_email` -> `customer.email`
- `customer_name` -> `customer.name`
- `uuid` -> `customer.id`
- `contexts` -> `contexts`
- `webhook` -> `webhook_url`
- `mcp_servers`、`http_tools`、`knowledge_config` 原样透传

### 4.2 CustomerServiceState

`CustomerServiceState` 是 harness 和工具共享的结构化状态：

```python
CustomerServiceState:
  request_id
  tenant_id
  channel
  subject
  content
  customer
  contexts

  current_intent
  slot_schema
  collected_slots
  missing_required_slots

  selected_template
  template_variables
  retrieved_knowledge
  user_memories

  order_result
  logistics_result
  http_tool_results
  mcp_tool_results
  tool_call_history

  need_handoff_to_human
  handoff_type
  handoff_reason

  final_reply
  token_usage
  performance_stats
  events
```

状态模型不为每个行业写死字段。订单号、产品型号、宠物类型、错误码、假发颜色等业务信息统一进入 `collected_slots`。

### 4.3 CustomerServiceResponse

统一响应：

```python
CustomerServiceResponse:
  request_id
  status
  reply:
    subject
    body
  need_handoff_to_human
  handoff_type
  handoff_reason
  agent_chain
  token_usage
  processing_time
  error
  state_snapshot
```

异步模式立即返回：

```python
AcceptedResponse:
  status = "accepted"
  request_id
  accepted_at
  message
```

后台执行完成后将 `CustomerServiceResponse` POST 到 `webhook_url`。

## 5. Agent Harness 执行流程

核心入口是 `CustomerServiceHarness.run()`：

```text
CustomerServiceRequest
  -> init CustomerServiceState
  -> start trace / performance tracker
  -> prepare tools
  -> prepare MCP servers
  -> pre-retrieve memory
  -> assemble instructions
  -> build user message
  -> execute OpenAI Agent
  -> post-process reply
  -> apply handoff policy
  -> reduce state into response
  -> webhook / return response
```

`BusinessAgentExecutor` 创建并执行 OpenAI Agent：

```python
Agent[CustomerServiceState](
    name="Customer Service Agent",
    instructions=instructions,
    model=model,
    tools=tools,
    mcp_servers=mcp_servers,
)
```

并调用：

```python
Runner.run(
    starting_agent=agent,
    input=user_message,
    context=state,
    hooks=hooks,
    max_turns=max_turns,
)
```

执行前必须设置当前请求状态上下文，保证工具能写回同一个 `CustomerServiceState`。第一版以 OpenAI Agents SDK 的 `context` 为主进行状态传递，并保留 contextvars 作为工具层兼容机制，避免动态工具和普通函数工具拿不到当前请求状态。

## 6. 工具体系

### 6.1 核心工具

默认注册：

- `extract_slots`：配置驱动的 slot 抽取和状态写入工具。
- `get_rag_knowledge`：企业知识库检索工具。
- `handoff_to_human`：转人工决策写入工具。

可按请求启用：

- `check_order_status`
- `check_shipping_status`
- `retrieve_memory`

模板选择不作为默认工具，而由后端策略处理：

- `TemplateRepository`
- `TemplatePolicy`
- `ReplyPostProcessor`

### 6.2 extract_slots

`extract_slots` 不写死订单号、SKU、地址等字段，而是根据当前租户或请求的 `slot_schema` 工作。

示例 `slot_schema`：

```json
[
  {
    "name": "order_id",
    "description": "客户订单号",
    "type": "string",
    "required": false,
    "aliases": ["订单号", "order number", "order id"]
  },
  {
    "name": "product_model",
    "description": "产品型号",
    "type": "string",
    "required": false,
    "aliases": ["model", "型号"]
  },
  {
    "name": "symptom",
    "description": "故障现象或客户遇到的问题",
    "type": "string",
    "required": true
  }
]
```

Agent 调用 `extract_slots` 时传入：

```json
{
  "slots": [
    {
      "name": "product_model",
      "value": "Airdog X5",
      "confidence": 0.88,
      "source": "current_message"
    },
    {
      "name": "symptom",
      "value": "will not turn on",
      "confidence": 0.82,
      "source": "current_message"
    }
  ]
}
```

工具职责：

- 校验 slot 是否在 schema 中。
- 标准化 slot name。
- 写入 `state.collected_slots`。
- 计算 `missing_required_slots`。
- 记录 `tool_call_history`。

### 6.3 动态 HTTP 工具

请求可传 `http_tools` 配置，系统动态生成 Agent 工具。

支持：

- 工具名、描述、参数 schema。
- 必填参数校验。
- GET/POST/PUT/DELETE/PATCH。
- headers。
- timeout。
- retry/backoff。
- response_mapping。
- 结果缓存。
- 工具调用事件和 Langfuse span。

测试时提供本地 mock HTTP server，例如：

- `/mock/order`
- `/mock/logistics`
- `/mock/refund-policy`

### 6.4 MCP 工具

内置本地 `stdio` MCP 示例 server：

`mcp_servers/product_support_server.py`

示例工具：

- `lookup_product_manual(product_or_sku, question)`
- `check_warranty_policy(product_or_sku)`

`MCPManager` 支持：

- `stdio`
- `sse`
- `streamable_http`
- 工具白名单/黑名单
- 工具描述覆盖
- 参数描述覆盖
- 连接失败降级
- 生命周期清理

## 7. RAG、Memory、模板和多语言

### 7.1 RAG

第一版提供：

- `MockKnowledgeRetriever`：默认启用，读取 `data/knowledge/*.json`。
- `RemoteKnowledgeRetriever`：接口预留，按配置请求远程 RAG 服务。

统一检索参数：

```python
tenant_id
query
kb_ids
file_ids
top_k
threshold
```

`knowledge_config` 支持：

```json
{
  "knowledges": [
    {
      "kbId": "kb_after_sales",
      "fileIds": ["return_policy", "warranty_policy"],
      "isAll": false,
      "kbName": "售后知识库"
    }
  ]
}
```

第一版不做用户问题到 `fileIds` 的自动路由。`fileIds/kbIds` 由请求、demo case 或租户配置提供。后续可增加 `KnowledgeRouter`。

### 7.2 Memory

第一版使用 `MockMemoryService`：

- 从 `data/mock_memories.json` 按 `customer.id` 读取记忆。
- 根据用户问题做简单相关性匹配。
- 返回 top memories。
- 高相关记忆注入 instructions。

当请求启用 `retrieve_memory` 工具时，Agent 可以主动查询记忆。

### 7.3 模板

模板由后端策略处理：

- `TemplateRepository` 从 `data/reply_templates.json` 读取模板。
- `TemplatePolicy` 根据 intent、handoff_type、channel、language、工具结果选择模板。
- `ReplyPostProcessor` 清理模型输出，必要时用模板兜底。

模板用于规范语气、结构和兜底，不直接覆盖有效模型回复。

### 7.4 多语言

`PromptAssembler` 要求 Agent 使用用户实际语言回复，并保持称呼、正文、签名语言一致。

第一版至少支持：

- English
- Deutsch
- Francais
- Espanol
- 中文

语言识别先使用简单规则或交给 Agent 判断，不引入复杂语言检测依赖。

## 8. 转人工策略

第一版采用 Agent 工具决策 + 后端策略校验。

Agent 通过 `handoff_to_human` 写入：

```python
need_handoff_to_human: bool
handoff_type: Literal["no_handoff", "reply_handoff", "no_reply_handoff"]
handoff_reason: str
```

`HandoffPolicy` 在 Agent 结束后做确定性校验：

- 回复为空或过短 -> `reply_handoff`
- 最大执行轮次超限 -> `reply_handoff`
- 核心依赖失败且无法回答 -> `reply_handoff`
- 用户明确要求人工客服 -> `reply_handoff` 或 `no_reply_handoff`
- 情绪激烈、投诉升级、法律/安全/高额赔偿 -> `no_reply_handoff`
- 状态不一致时自动修正

如果最终为 `no_reply_handoff`，`StateReducer` 清空 `reply.body`。

## 9. API、异步处理和观测

### 9.1 API

提供：

- `POST /customer-service/respond`
- `POST /emailv4`
- `GET /health`
- `GET /demo/cases`

### 9.2 异步处理

同步模式：

```text
请求 -> harness.run() -> CustomerServiceResponse
```

异步模式：

```text
请求 -> AcceptedResponse
     -> background task 执行 harness.run()
     -> webhook_url 回调 CustomerServiceResponse
```

webhook 失败时第一版只记录事件，不做重试队列。后续可接入 Celery、RQ、Kafka 或 Redis Stream。

### 9.3 Observability

默认记录本地结构化事件：

- request_start
- tool_registry_prepare
- mcp_connect
- memory_retrieve
- rag_retrieve
- dynamic_http_call
- agent_start
- agent_end
- tool_start
- tool_end
- post_process
- handoff_policy
- response_built
- request_error

如果存在 Langfuse 配置，则额外创建 trace/span。没有配置时不影响本地运行。

实现 OpenAI Agents SDK hooks：

- Agent start/end
- Tool start/end
- Handoff event：当 SDK 事件可用时记录
- Usage 记录

## 10. 错误处理

统一策略：

- 单个 MCP server 连接失败：降级，不阻断整体请求。
- 单个动态 HTTP 工具失败：返回结构化错误给 Agent。
- RAG 失败：返回空知识或 fallback 提示。
- OpenAI Agent 执行失败：返回 error response，并可标记转人工。
- 最大轮次超限：尽量提取部分结果，否则兜底回复并转人工。
- webhook 失败：记录事件，不影响已完成处理结果。

## 11. 测试与演示

### 11.1 自动化测试

覆盖：

- 请求模型和 `/emailv4` 兼容映射。
- `CustomerServiceState` 初始化和状态更新。
- `extract_slots` 根据不同 `slot_schema` 写入不同字段。
- RAG 按 `kbIds/fileIds/top_k/threshold` 过滤。
- mock memory 注入。
- 动态 HTTP 工具参数校验、响应映射、缓存、重试。
- `HandoffPolicy` 一致性修正和兜底。
- `ReplyPostProcessor` 清理 THINK/ACTION/OBSERVE。
- 一个端到端 harness 流程，必要时 mock OpenAI Agent 结果，避免测试强依赖真实 API。

### 11.2 演示脚本

提供：

```text
scripts/run_demo_cases.py
scripts/run_mock_http_server.py
```

`run_demo_cases.py` 打印：

- 用户输入
- Agent 回复
- 工具调用摘要
- RAG 命中
- 是否转人工
- token usage / processing time
- trace/event 摘要

### 11.3 Demo 场景

内置 10 个场景：

1. 英文订单查询：提取 `order_id`，调用订单工具，回复订单状态。
2. 物流延迟：调用物流工具，结合售后知识库回复。
3. 退货政策咨询：按 `kbId/fileIds` 命中退货政策。
4. 3C 故障排障：slot schema 抽取 `product_model`、`error_code`、`symptom`，调用 RAG/MCP。
5. 宠物用品咨询：抽取 `pet_type`、`pet_weight`，回答产品适配建议。
6. 假发商品推荐：抽取 `hair_type`、`color`、`length`，调用知识库或 MCP。
7. 客户明确要求人工：`reply_handoff`。
8. 情绪激烈投诉/高风险赔偿：`no_reply_handoff`。
9. 德语邮件回复：验证多语言称呼、正文、签名一致。
10. 动态 HTTP 工具演示：通过配置生成售后查询工具并调用本地 mock server。

## 12. 生产演进路径

第一版 provider 可替换为生产实现：

- `MockKnowledgeRetriever` -> `RemoteKnowledgeRetriever` / 向量库 / 企业知识库 API。
- `MockMemoryService` -> Redis / PostgreSQL / 专用 memory API。
- mock order/logistics -> 企业订单和物流系统。
- `StaticTemplateRepository` -> 数据库模板配置。
- 本地事件日志 -> Langfuse / OpenTelemetry / ELK。
- 代码内 HandoffPolicy -> 租户级规则配置和工单系统。
- FastAPI background task -> 消息队列和任务系统。
- 本地 MCP 示例 -> 多个企业 MCP server。

关键原则：核心 harness 只依赖抽象接口，不依赖 mock 数据源。生产演进应替换 provider，而不是重写主流程。
