# Phase 6 Live Model and End-to-End Validation Design

## 1. Purpose

Phase 1 through Phase 5 established a near-production intelligent customer
service backend with FastAPI, `CustomerServiceHarness`, OpenAI Agents SDK tool
execution, tenant-scoped knowledge and memory, MCP, dynamic HTTP tools, handoff
policy, response post-processing, and observability.

Phase 6 adds two missing validation layers:

1. A deterministic offline end-to-end suite that exercises the complete
   application chain without consuming model API quota.
2. Explicitly enabled live-model smoke and tool-use tests against an
   OpenAI-compatible provider.

Phase 6 also introduces a provider-neutral model configuration boundary for
OpenAI, DeepSeek, and DashScope. The Harness and business tools must remain
unaware of the selected provider.

## 2. Scope

Phase 6 will:

- Add an OpenAI-compatible provider configuration resolver.
- Support one globally selected provider per application process.
- Support OpenAI, DeepSeek, and DashScope configurations in the same `.env`.
- Create a local `.env` for secrets and a tracked `.env.example` template.
- Build the selected provider's `AsyncOpenAI` client and
  `OpenAIChatCompletionsModel` adapter.
- Keep provider selection outside request payloads.
- Add deterministic offline full-chain E2E tests.
- Add opt-in live-model connectivity and tool-use tests.
- Exercise real model reasoning against local mock order, logistics, RAG, MCP,
  and dynamic HTTP capabilities.
- Document commands for selecting and testing each provider.

Phase 6 will not:

- Connect to production order, logistics, refund, or after-sales systems.
- Add request-level provider switching.
- Add tenant-to-provider routing.
- Run live-model tests during the default test suite.
- Compare or rank providers automatically.
- Assert exact generated wording from a live model.
- Add durable job queues or replace FastAPI background tasks.
- Upgrade the current single-Agent Harness into specialist agents.

## 3. Configuration Contract

The application will accept the following environment fields:

```dotenv
MODEL_PROVIDER=openai
RUN_LIVE_MODEL_TESTS=0

OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=

DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=

MODEL_TIMEOUT_SECONDS=60
MODEL_MAX_RETRIES=1
```

`MODEL_PROVIDER` must be one of:

- `openai`
- `deepseek`
- `dashscope`

Only the selected provider's API key, base URL, and model are required at
runtime. Unselected provider credentials may remain empty.

`.env` and `.env.example` will contain the same field names. `.env` remains
ignored by Git and is intended for local secrets. `.env.example` is committed
with empty credentials and safe endpoint examples.

Provider credentials must never be accepted from an API request, returned in a
response, or included in trace metadata.

## 4. Provider Boundary

### 4.1 ProviderConfig

A typed `ProviderConfig` value will contain:

- provider name
- API key
- base URL
- model name
- request timeout
- maximum SDK retries

The value is constructed from application settings and represents only the
currently selected provider.

### 4.2 ProviderConfigResolver

`ProviderConfigResolver` will:

1. Validate `MODEL_PROVIDER`.
2. Select the matching provider-prefixed settings.
3. Validate that API key, base URL, and model name are non-empty.
4. Validate that the base URL uses `http` or `https`.
5. Return a sanitized configuration error that names missing fields without
   exposing configured values.

Provider resolution happens before a model request is made. Invalid
configuration therefore fails deterministically without consuming model quota.

### 4.3 Model Adapter Factory

The selected configuration will be converted into:

```text
AsyncOpenAI(
  api_key=<selected key>,
  base_url=<selected base URL>,
  timeout=<configured timeout>,
  max_retries=<configured retries>
)
  -> OpenAIChatCompletionsModel(
       model=<selected model>,
       openai_client=<provider client>
     )
```

`BusinessAgentExecutor` will pass this model adapter to `Agent`. The existing
Harness, prompt assembler, tools, MCP manager, state reducer, handoff policy,
and API routes will not branch on provider names.

The Chat Completions adapter is the common compatibility baseline for the
three providers. A future provider may use a different adapter behind the same
factory boundary if its protocol requires it.

### 4.4 Tracing

Existing local tracing and optional Langfuse tracing remain active.

For non-OpenAI providers, the Agents SDK must not attempt to send traces to
OpenAI using the selected third-party API key. Provider-neutral local trace
events remain available.

For OpenAI, model execution and application tracing remain separate concerns.
Phase 6 does not require OpenAI-hosted tracing to pass live tests.

## 5. Model Selection Rules

The active default model comes from the selected provider's model field.

The existing request-level `model` field may override the active model only
within an explicit allowlist for the current application process. It cannot
change API key, base URL, or provider.

If no request-level model is supplied, the selected provider model is used.
If a request supplies a model outside the configured allowlist, the Harness
returns its existing sanitized error response before invoking the provider.

`ALLOWED_MODELS` remains a comma-separated allowlist. Deployment configuration
must include the selected provider's default model in that list.

## 6. Error Handling

The provider layer will distinguish:

- unsupported provider
- missing API key
- missing base URL
- missing model name
- invalid base URL
- authentication failure
- rate limiting
- timeout
- protocol or response incompatibility
- other provider execution failure

Application responses continue to use the existing customer-safe error:

```text
Customer service processing failed.
```

Detailed diagnostics are available only to tests and sanitized logs or traces.
Diagnostics may contain provider name, exception class, HTTP status, and
request ID, but never authorization headers, API keys, or complete upstream
response bodies.

Live tests must not automatically repeat a failed scenario outside the SDK's
configured retry limit. This bounds cost and avoids multiplying failures.

## 7. Offline End-to-End Validation

The offline E2E suite will exercise:

```text
FastAPI TestClient
  -> request validation
  -> API dependency
  -> CustomerServiceHarness
  -> tool and MCP preparation
  -> deterministic model runner
  -> tool execution
  -> state updates
  -> reply post-processing
  -> handoff policy
  -> StateReducer
  -> HTTP response
```

Unlike isolated route tests, the offline E2E tests will not replace the
Harness. Unlike live tests, they will replace only the nondeterministic model
boundary.

Required offline scenarios:

- plain customer-service reply
- order status lookup
- logistics lookup
- RAG retrieval with `kbId` and `fileIds`
- human-requested escalation
- high-risk no-reply escalation
- multilingual response post-processing
- local MCP product-manual lookup
- trusted dynamic HTTP tool execution
- tenant isolation for business data and knowledge

These tests run as part of the default `pytest` suite and must not require
network access or credentials.

## 8. Live-Model Validation

### 8.1 Execution Gate

Live tests run only when:

```dotenv
RUN_LIVE_MODEL_TESTS=1
```

When the variable is absent or false, every live test is reported as skipped.
The default test command therefore remains deterministic and free of provider
charges.

Live tests use only the provider selected by `MODEL_PROVIDER`. Testing another
provider requires changing `MODEL_PROVIDER` and running the suite again.

### 8.2 Layer 1: Connectivity

The first live test validates:

- provider configuration resolves
- the model can return a non-empty customer-service response
- the Harness reports success
- token usage is present when supplied by the provider
- Agent lifecycle trace events are recorded
- no credential appears in the serialized response or trace snapshot

### 8.3 Layer 2: Core Tool Use

Separate live tests validate model-selected use of:

- `check_order_status`
- `check_shipping_status`
- `get_rag_knowledge`
- `handoff_to_human`

Each test provides a prompt containing enough identifying information for the
model to choose the intended tool. Assertions target observable contracts:

- expected tool name appears in `tool_call_history`
- required argument values are correct
- returned business facts match local mock data
- relevant state fields are updated
- the final response does not contradict tool results

Tests do not require exact sentences or punctuation.

### 8.4 Layer 3: Extended Integrations

MCP and dynamic HTTP live tests are independent from the core tool-use tests:

- The MCP test starts the trusted local stdio server and checks that a product
  manual tool result is recorded.
- The dynamic HTTP test uses a trusted local mock endpoint registered by
  deployment-side test configuration.

These tests never call production business systems.

### 8.5 Cost and Stability Controls

- Use a small number of representative prompts.
- Keep `max_turns` low for live cases.
- Do not run multiple providers in one invocation.
- Do not retry entire pytest cases automatically.
- Mark live tests clearly so they can be selected independently.
- Serialize live cases by default to reduce rate-limit pressure.

## 9. Test Organization

Phase 6 introduces focused test modules:

```text
tests/
  e2e/
    test_customer_service_e2e.py
    test_live_model_smoke.py
    test_live_model_tools.py
  test_model_provider.py
```

Test markers:

```ini
e2e: deterministic full-chain application tests
live_model: tests that call the configured external model provider
```

Expected commands:

```bash
# Existing unit, integration, and offline E2E tests
.venv/bin/python -m pytest -v

# Offline full-chain tests only
.venv/bin/python -m pytest -m e2e -v

# Selected provider live tests
RUN_LIVE_MODEL_TESTS=1 \
  .venv/bin/python -m pytest -m live_model -v
```

## 10. Documentation Updates

The README will document:

- `.env.example` setup
- provider selection
- OpenAI, DeepSeek, and DashScope configuration fields
- the distinction between offline E2E and live-model tests
- exact test commands
- cost and credential warnings
- the fact that live-model tests use local mock business systems

`AGENTS.md` will be updated so future work uses the provider configuration
boundary and knows that Phase 6 live tests are opt-in.

The existing five-phase implementation plan remains historical evidence of
Phase 1 through Phase 5. Phase 6 receives its own implementation plan and
completion checkpoint.

## 11. Acceptance Criteria

Phase 6 is complete when:

1. `.env` and `.env.example` contain the agreed provider fields.
2. `.env` remains ignored and no secret is committed.
3. OpenAI, DeepSeek, and DashScope configurations resolve through one typed
   provider boundary.
4. The executor uses an `OpenAIChatCompletionsModel` built from the selected
   provider's OpenAI-compatible client.
5. Invalid or incomplete provider configuration fails before a model call and
   returns sanitized diagnostics.
6. Default pytest runs include offline full-chain E2E coverage and make no
   external model calls.
7. Live tests are skipped unless `RUN_LIVE_MODEL_TESTS=1`.
8. Live connectivity and core tool-use cases can run against the selected
   provider.
9. MCP and dynamic HTTP live cases use only trusted local test integrations.
10. Responses, logs, and trace snapshots do not expose provider credentials.
11. Existing Phase 1 through Phase 5 tests continue to pass.

