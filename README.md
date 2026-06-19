# OpenAI Agents Intelligent Customer Service

Near-production intelligent customer service backend demo built with
`openai-agents-python`, FastAPI, and a lightweight Agent Harness.

The first version uses one customer-service Agent with request-scoped tools,
tenant-scoped memory and knowledge, MCP integrations, handoff policy, async
webhook delivery, and structured observability. Mock providers are isolated
behind replaceable boundaries for later production integration.

## Capabilities

- OpenAI Agents SDK `Agent` and `Runner.run()` execution
- Structured multi-turn customer-service state
- Config-driven slot extraction
- Tenant-scoped mock RAG with `kbId` and `fileIds` filtering
- Tenant-scoped long-term memory pre-retrieval
- Dynamic HTTP tools with validation, retry, caching, redaction, and SSRF controls
- Trusted MCP server registry with stdio, SSE, and streamable HTTP support
- Tool filtering and MCP description overrides
- Handoff policy for empty replies, human requests, complaints, and safety risks
- FastAPI sync responses and async webhook callbacks
- Local structured tracing and OpenAI Agents SDK lifecycle hooks
- Optional Langfuse initialization

## Environment

Use the project-local virtual environment for every command:

```bash
cd /Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service
/opt/miniconda3/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Optional environment variables:

```bash
export OPENAI_API_KEY="..."
export ALLOWED_MODELS="gpt-4.1-mini"
export DYNAMIC_HTTP_ALLOWED_HOSTS="api.example.com"
export WEBHOOK_ALLOWED_HOSTS="hooks.example.com"

# Optional observability
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

`DYNAMIC_HTTP_ALLOWED_HOSTS` and `WEBHOOK_ALLOWED_HOSTS` are comma-separated
exact hostnames. Production defaults are fail-closed when allowlists are absent.
`ALLOWED_MODELS` limits request-selectable models. Requests also constrain
`max_turns` to the range `1..20`.

## Run The API

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Synchronous request:

```bash
curl -X POST http://127.0.0.1:8000/customer-service/respond \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "tenant_a",
    "content": "What is your return policy?",
    "async_mode": false
  }'
```

The default execution path calls the configured OpenAI model and therefore
requires `OPENAI_API_KEY`. Use the offline demo below when no credentials are
available.

Async mode requires both `webhook_url` and a matching
`WEBHOOK_ALLOWED_HOSTS` entry. It returns HTTP `202` and posts the completed
`CustomerServiceResponse` to the webhook.

## Run Offline Demo Cases

The demo injects a deterministic fake runner and makes no OpenAI or network
calls:

```bash
.venv/bin/python scripts/run_demo_cases.py
```

It prints one compact JSON object per line for ten cases: order, logistics,
return policy, electronics troubleshooting with MCP, pet product advice, wig
recommendation, human handoff, high-risk escalation, German reply, and dynamic
HTTP tool execution.

## Run Mock Business APIs

Port `8765` is the default. Override it when already occupied:

```bash
MOCK_HTTP_PORT=18765 .venv/bin/python scripts/run_mock_http_server.py
```

Available endpoints:

- `POST /mock/order`
- `POST /mock/logistics`
- `GET /mock/refund-policy`

Example:

```bash
curl -X POST http://127.0.0.1:18765/mock/order \
  -H 'Content-Type: application/json' \
  -d '{"order_id": "A100"}'
```

Dynamic HTTP definitions are registered by deployment code per tenant. Request
payloads may select a trusted tool name but cannot override its URL, method,
headers, credentials, schema, or retry policy. Localhost access is suitable only
for development bindings and is rejected by production HTTPS/private-address
controls.

## MCP

The built-in `product_support` stdio MCP server is registered for `tenant_a` by
deployment code and can be selected by request name. Request payloads cannot
provide arbitrary executable commands, remote MCP URLs, environment variables,
or credentials. Other tenants fail closed unless their own trusted bindings are
registered.

```bash
.venv/bin/python mcp_servers/product_support_server.py
```

Production deployments should load trusted MCP definitions from a secured
tenant configuration service rather than accepting executable definitions from
API requests.

## Run Tests

```bash
.venv/bin/python -m pytest -v
```

Useful focused checks:

```bash
.venv/bin/python -m pytest tests/test_demo_script.py -v
.venv/bin/python -m pytest tests/test_mock_http_server.py -v
.venv/bin/python -m py_compile mcp_servers/product_support_server.py
```

## Project Structure

```text
app/api/             FastAPI routes and legacy payload compatibility
app/harness/         Agent execution, prompt, handoff, post-processing, reduction
app/observability/   Local tracing, performance metrics, SDK hooks
app/retrieval/       Knowledge and memory providers
app/state/           Request, state, and response models
app/tools/           Core tools, dynamic HTTP tools, MCP management
data/                Mock knowledge, memory, orders, logistics, templates
mcp_servers/         Local MCP example server
scripts/             Offline demo and mock business API server
tests/               Unit and integration tests
```

## Current Limitations

- FastAPI `BackgroundTasks` is in-process and not durable across restarts.
- Webhook delivery has no persistent retry queue.
- Mock RAG uses keyword overlap rather than embeddings or hybrid retrieval.
- Mock memory, order, logistics, and template data are local JSON files.
- Trusted dynamic HTTP and MCP definitions are currently registered in
  application code.
- Configured Langfuse credentials mirror local request/tool events and flush at
  response completion. Local trace events remain available when the dependency,
  configuration, or remote service is unavailable.
- The optional dependency currently targets the Langfuse Python v2 tracing API;
  migrating to its OpenTelemetry-native SDK is a separate production upgrade.
- The first version is a single-Agent Harness. Router, planner, and specialist
  Agent orchestration are future evolution paths, not current behavior.

## Design References

- `docs/superpowers/specs/2026-06-17-openai-agents-customer-service-design.md`
- `docs/superpowers/plans/2026-06-17-openai-agents-customer-service-demo-implementation-plan.md`
- `findings/single_react_agent_1118_source_findings.md`
