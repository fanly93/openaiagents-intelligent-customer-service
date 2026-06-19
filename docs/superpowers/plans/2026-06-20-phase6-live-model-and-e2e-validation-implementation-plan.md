# Phase 6 Live Model and End-to-End Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-neutral OpenAI-compatible model configuration, deterministic full-chain E2E coverage, and opt-in live-model validation for OpenAI, DeepSeek, and DashScope.

**Architecture:** Introduce a focused provider boundary that resolves the globally selected provider and creates an `AsyncOpenAI` plus `OpenAIChatCompletionsModel` adapter. Inject that adapter into the existing `BusinessAgentExecutor` without adding provider branches to the Harness. Validate the full application through real FastAPI and Harness paths, replacing only the model boundary for offline E2E tests and using local mock business integrations for live tests.

**Tech Stack:** Python 3.13, Pydantic Settings, OpenAI Python client, `openai-agents-python`, FastAPI, pytest, pytest-asyncio, httpx, local stdio MCP.

**Approved Spec:** `docs/superpowers/specs/2026-06-20-phase6-live-model-and-e2e-validation-design.md`

---

## Execution Structure

Phase 6 is divided into five implementation stages. Stop after every stage,
run the listed regression checks, update this plan's checkboxes, commit the
stage, and report completion before proceeding.

Parallel-agent assessment:

- Stage 1 is sequential because `.env`, settings, provider resolution, and
  provider tests define one shared contract.
- Stage 2 is sequential because executor integration depends on Stage 1 and
  modifies shared executor and Harness behavior.
- Stage 3 is sequential because the E2E fixture and scenarios share the same
  test harness and dependency overrides.
- Stage 4 can split live smoke and live tool test authoring after the shared
  live fixtures exist, provided agents edit different test files.
- Stage 5 documentation and final verification are sequential because they
  reconcile all implemented behavior.

## File Map

**Create**

- `.env` - ignored local credentials and provider selection.
- `.env.example` - tracked safe provider configuration template.
- `app/providers/__init__.py` - provider package exports.
- `app/providers/openai_compatible.py` - provider resolution, validation,
  client construction, model adapter creation, and sanitized provider errors.
- `tests/test_model_provider.py` - provider configuration and adapter tests.
- `tests/e2e/__init__.py` - E2E test package.
- `tests/e2e/conftest.py` - deterministic Runner boundary and real Harness/API
  fixtures.
- `tests/e2e/test_customer_service_e2e.py` - offline full-chain scenarios.
- `tests/e2e/test_live_model_smoke.py` - opt-in provider connectivity test.
- `tests/e2e/test_live_model_tools.py` - opt-in live tool, MCP, and dynamic HTTP
  scenarios.

**Modify**

- `app/config.py` - expose all provider and live-test settings.
- `app/harness/business_agent_executor.py` - use the provider model factory.
- `app/harness/customer_service_harness.py` - use selected default model for
  validation and record sanitized provider diagnostics.
- `pyproject.toml` - register `e2e` and `live_model` pytest markers.
- `tests/test_harness_flow.py` - update executor contract tests.
- `README.md` - provider setup and test commands.
- `AGENTS.md` - Phase 6 provider and live-test conventions.

---

## Stage 1: Provider Configuration Contract

### Task 1.1: Create Environment Templates

**Files:**

- Create: `.env`
- Create: `.env.example`
- Verify: `.gitignore`

- [ ] **Step 1: Confirm `.env` remains ignored**

Run:

```bash
git check-ignore -v .env
```

Expected: output identifies the `.env` rule from `.gitignore`.

- [ ] **Step 2: Create the local `.env`**

Create `.env` with no real credentials:

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
ALLOWED_MODELS=

ENVIRONMENT=local
RAG_MODE=mock
DYNAMIC_HTTP_ALLOWED_HOSTS=
WEBHOOK_ALLOWED_HOSTS=

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

- [ ] **Step 3: Create the tracked `.env.example`**

Create `.env.example` with the same safe content as `.env`. Keep all API key
values empty.

- [ ] **Step 4: Verify no local secret file is staged**

Run:

```bash
git status --short
git check-ignore .env
```

Expected: `.env.example` is untracked, `.env` is absent from `git status`, and
`git check-ignore` exits successfully.

### Task 1.2: Add Provider Settings and Resolver

**Files:**

- Modify: `app/config.py`
- Create: `app/providers/__init__.py`
- Create: `app/providers/openai_compatible.py`
- Test: `tests/test_model_provider.py`

- [ ] **Step 1: Write failing settings and resolver tests**

Create `tests/test_model_provider.py`:

```python
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.providers.openai_compatible import (
    ProviderConfigurationError,
    resolve_provider_config,
)


def provider_settings(**updates):
    values = {
        "model_provider": "openai",
        "openai_api_key": "openai-secret",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-test",
        "deepseek_api_key": "deepseek-secret",
        "deepseek_base_url": "https://api.deepseek.com/v1",
        "deepseek_model": "deepseek-chat",
        "dashscope_api_key": "dashscope-secret",
        "dashscope_base_url": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "dashscope_model": "qwen-test",
        "model_timeout_seconds": 60.0,
        "model_max_retries": 1,
        "allowed_models": "gpt-test,deepseek-chat,qwen-test",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_settings_read_all_openai_compatible_provider_fields(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    settings = Settings(_env_file=None)

    assert settings.model_provider == "deepseek"
    assert settings.deepseek_api_key == "secret"
    assert settings.deepseek_model == "deepseek-chat"


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("openai", "gpt-test"),
        ("deepseek", "deepseek-chat"),
        ("dashscope", "qwen-test"),
    ],
)
def test_resolver_selects_only_the_active_provider(provider, expected_model):
    config = resolve_provider_config(
        provider_settings(model_provider=provider)
    )

    assert config.provider == provider
    assert config.model == expected_model
    assert config.timeout_seconds == 60.0
    assert config.max_retries == 1


def test_resolver_rejects_unsupported_provider_without_exposing_keys():
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(
            provider_settings(model_provider="unknown")
        )

    assert exc_info.value.code == "unsupported_provider"
    assert "openai-secret" not in str(exc_info.value)


def test_resolver_reports_missing_active_fields_only():
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(
            provider_settings(
                model_provider="deepseek",
                deepseek_api_key="",
            )
        )

    assert exc_info.value.code == "missing_provider_fields"
    assert exc_info.value.fields == ("DEEPSEEK_API_KEY",)


def test_resolver_rejects_non_http_base_url():
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_config(
            provider_settings(openai_base_url="file:///tmp/provider")
        )

    assert exc_info.value.code == "invalid_base_url"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_provider.py -v
```

Expected: collection fails because `app.providers.openai_compatible` does not
exist.

- [ ] **Step 3: Add provider settings**

Extend `Settings` in `app/config.py` with:

```python
    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")
    run_live_model_tests: bool = Field(
        default=False,
        alias="RUN_LIVE_MODEL_TESTS",
    )

    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")

    deepseek_api_key: str | None = Field(
        default=None,
        alias="DEEPSEEK_API_KEY",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str | None = Field(
        default=None,
        alias="DEEPSEEK_MODEL",
    )

    dashscope_api_key: str | None = Field(
        default=None,
        alias="DASHSCOPE_API_KEY",
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_model: str | None = Field(
        default=None,
        alias="DASHSCOPE_MODEL",
    )

    model_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        alias="MODEL_TIMEOUT_SECONDS",
    )
    model_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        alias="MODEL_MAX_RETRIES",
    )
```

Remove the old duplicate `openai_api_key` declaration. Keep `default_model`
temporarily for backward compatibility until Stage 2 updates executor usage.

- [ ] **Step 4: Implement the resolver**

Create `app/providers/__init__.py`:

```python
"""OpenAI-compatible model provider integration."""
```

Create `app/providers/openai_compatible.py`:

```python
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


SUPPORTED_PROVIDERS = ("openai", "deepseek", "dashscope")


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int


class ProviderConfigurationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        provider: str | None = None,
        fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.fields = fields


def resolve_provider_config(settings: Any) -> ProviderConfig:
    provider = str(settings.model_provider).strip().casefold()
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderConfigurationError(
            "unsupported_provider",
            f"Unsupported model provider: {provider or '<empty>'}.",
            provider=provider or None,
        )

    prefix = provider.upper()
    values = {
        f"{prefix}_API_KEY": getattr(settings, f"{provider}_api_key", None),
        f"{prefix}_BASE_URL": getattr(settings, f"{provider}_base_url", None),
        f"{prefix}_MODEL": getattr(settings, f"{provider}_model", None),
    }
    missing = tuple(
        name for name, value in values.items()
        if not str(value or "").strip()
    )
    if missing:
        raise ProviderConfigurationError(
            "missing_provider_fields",
            "Selected model provider configuration is incomplete: "
            + ", ".join(missing),
            provider=provider,
            fields=missing,
        )

    base_url = str(values[f"{prefix}_BASE_URL"]).strip()
    if urlparse(base_url).scheme not in {"http", "https"}:
        raise ProviderConfigurationError(
            "invalid_base_url",
            "Selected model provider base URL must use http or https.",
            provider=provider,
            fields=(f"{prefix}_BASE_URL",),
        )

    return ProviderConfig(
        provider=provider,
        api_key=str(values[f"{prefix}_API_KEY"]).strip(),
        base_url=base_url.rstrip("/"),
        model=str(values[f"{prefix}_MODEL"]).strip(),
        timeout_seconds=float(settings.model_timeout_seconds),
        max_retries=int(settings.model_max_retries),
    )
```

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_provider.py -v
```

Expected: all provider settings and resolver tests pass.

- [ ] **Step 6: Run Stage 1 regression**

Run:

```bash
.venv/bin/python -m pytest tests/test_state_models.py tests/test_security_boundaries.py -v
```

Expected: all selected regression tests pass.

- [ ] **Step 7: Commit Stage 1**

Run:

```bash
git add .env.example app/config.py app/providers tests/test_model_provider.py
git commit -m "feat: add openai-compatible provider configuration"
```

Do not add `.env`.

- [ ] **Stage 1 Checkpoint: stop and report**

Report provider fields, resolver behavior, tests run, commit hash, and confirm
that `.env` is ignored and contains no filled credentials.

---

## Stage 2: Agents SDK Model Adapter Integration

### Task 2.1: Build the OpenAI-Compatible Model Factory

**Files:**

- Modify: `app/providers/openai_compatible.py`
- Test: `tests/test_model_provider.py`

- [ ] **Step 1: Write failing model factory tests**

Append to `tests/test_model_provider.py`:

```python
from app.providers.openai_compatible import OpenAICompatibleModelFactory


def test_model_factory_builds_chat_completions_adapter(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    class FakeModel:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

    monkeypatch.setattr(
        "app.providers.openai_compatible.AsyncOpenAI",
        FakeClient,
    )
    monkeypatch.setattr(
        "app.providers.openai_compatible.OpenAIChatCompletionsModel",
        FakeModel,
    )

    factory = OpenAICompatibleModelFactory(
        settings=provider_settings(model_provider="deepseek")
    )
    model = factory.create()

    assert isinstance(model, FakeModel)
    assert captured["client"] == {
        "api_key": "deepseek-secret",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 60.0,
        "max_retries": 1,
    }
    assert captured["model"]["model"] == "deepseek-chat"


def test_model_factory_allows_only_allowlisted_override():
    factory = OpenAICompatibleModelFactory(
        settings=provider_settings(model_provider="openai"),
        allowed_models={"gpt-test", "gpt-test-2"},
    )

    assert factory.resolve_model_name("gpt-test-2") == "gpt-test-2"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        factory.resolve_model_name("deepseek-chat")

    assert exc_info.value.code == "model_not_allowed"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_model_provider.py::test_model_factory_builds_chat_completions_adapter \
  tests/test_model_provider.py::test_model_factory_allows_only_allowlisted_override \
  -v
```

Expected: import fails because `OpenAICompatibleModelFactory` does not exist.

- [ ] **Step 3: Implement the factory**

Add to `app/providers/openai_compatible.py`:

```python
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI


class OpenAICompatibleModelFactory:
    def __init__(
        self,
        settings: Any,
        allowed_models: set[str] | None = None,
    ) -> None:
        self.settings = settings
        self.config = resolve_provider_config(settings)
        configured = allowed_models or {
            item.strip()
            for item in str(settings.allowed_models).split(",")
            if item.strip()
        }
        self.allowed_models = configured or {self.config.model}
        self.allowed_models.add(self.config.model)

    def resolve_model_name(self, requested_model: str | None = None) -> str:
        model = requested_model or self.config.model
        if model not in self.allowed_models:
            raise ProviderConfigurationError(
                "model_not_allowed",
                "Requested model is not allowed for the active provider.",
                provider=self.config.provider,
                fields=("ALLOWED_MODELS",),
            )
        return model

    def create(self, requested_model: str | None = None) -> Any:
        model_name = self.resolve_model_name(requested_model)
        set_tracing_disabled(self.config.provider != "openai")
        client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        return OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        )
```

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_provider.py -v
```

Expected: all provider tests pass.

### Task 2.2: Inject the Factory into BusinessAgentExecutor

**Files:**

- Modify: `app/harness/business_agent_executor.py`
- Modify: `app/harness/customer_service_harness.py`
- Modify: `tests/test_harness_flow.py`

- [ ] **Step 1: Replace the old SDK model assertion with a failing adapter test**

Update
`test_business_agent_executor_uses_sdk_model_and_aggregate_usage` in
`tests/test_harness_flow.py`:

```python
@pytest.mark.asyncio
async def test_business_agent_executor_uses_model_factory_and_usage(monkeypatch):
    import agents

    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="chat",
        content="Hi",
    )
    adapter = object()
    requested_models = []

    class FakeFactory:
        def create(self, requested_model=None):
            requested_models.append(requested_model)
            return adapter

    async def fake_sdk_run(starting_agent, input, **kwargs):
        assert starting_agent.model is adapter
        assert input == "Hi"
        return SimpleNamespace(
            final_output="Hello from SDK",
            raw_responses=["response"],
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=7,
                    output_tokens=5,
                    total_tokens=12,
                )
            ),
        )

    monkeypatch.setattr(agents.Runner, "run", fake_sdk_run)

    result = await BusinessAgentExecutor(
        model_factory=FakeFactory()
    ).run(
        state=state,
        instructions="instructions",
        user_message="Hi",
        tools=[],
        mcp_servers=[],
        max_turns=3,
        model="gpt-test",
    )

    assert requested_models == ["gpt-test"]
    assert result.token_usage["total_tokens"] == 12
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_harness_flow.py::test_business_agent_executor_uses_model_factory_and_usage \
  -v
```

Expected: `BusinessAgentExecutor.__init__` rejects `model_factory`.

- [ ] **Step 3: Update BusinessAgentExecutor**

Change `BusinessAgentExecutor` to:

```python
from app.providers.openai_compatible import OpenAICompatibleModelFactory


class BusinessAgentExecutor:
    def __init__(
        self,
        runner: RunnerCallable | None = None,
        model_factory: Any | None = None,
    ) -> None:
        self._runner = runner
        self._model_factory = model_factory

    def _get_model_factory(self) -> Any:
        if self._model_factory is None:
            self._model_factory = OpenAICompatibleModelFactory(
                settings=get_settings()
            )
        return self._model_factory
```

Inside `_run_openai_agents`, replace the string model:

```python
        model_adapter = self._get_model_factory().create(model)
        agent = Agent[CustomerServiceState](
            name="Customer Service Agent",
            instructions=instructions,
            model=model_adapter,
            tools=tools,
            mcp_servers=mcp_servers,
        )
```

Remove `settings.default_model` from this execution path.

- [ ] **Step 4: Make Harness model validation provider-aware**

Replace `_validate_model` in `CustomerServiceHarness` with:

```python
    @staticmethod
    def _validate_model(model: str | None) -> None:
        if not model:
            return
        settings = get_settings()
        allowed_models = {
            item.strip()
            for item in settings.allowed_models.split(",")
            if item.strip()
        }
        if allowed_models and model not in allowed_models:
            raise ValueError("requested model is not allowed")
```

This preserves request-level protection while allowing the selected provider
factory to supply its default when `model` is absent.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_model_provider.py \
  tests/test_harness_flow.py \
  tests/test_security_boundaries.py \
  -v
```

Expected: all selected tests pass.

### Task 2.3: Normalize Provider Failures and Trace Safe Diagnostics

**Files:**

- Modify: `app/providers/openai_compatible.py`
- Modify: `app/harness/business_agent_executor.py`
- Modify: `app/harness/customer_service_harness.py`
- Modify: `tests/test_model_provider.py`
- Modify: `tests/test_harness_flow.py`

- [ ] **Step 1: Write failing provider error normalization tests**

Append to `tests/test_model_provider.py`:

```python
from app.providers.openai_compatible import (
    ProviderExecutionError,
    normalize_provider_exception,
)


@pytest.mark.parametrize(
    ("exception_name", "expected_code"),
    [
        ("AuthenticationError", "authentication_failed"),
        ("RateLimitError", "rate_limited"),
        ("APITimeoutError", "timeout"),
        ("APIStatusError", "protocol_error"),
        ("UnexpectedProviderFailure", "provider_execution_failed"),
    ],
)
def test_provider_exceptions_are_normalized_without_secret_messages(
    exception_name,
    expected_code,
):
    exception_type = type(exception_name, (Exception,), {})
    exc = exception_type("upstream leaked secret-value")
    exc.status_code = 401
    exc.request_id = "provider-request-id"

    normalized = normalize_provider_exception(exc, "deepseek")

    assert isinstance(normalized, ProviderExecutionError)
    assert normalized.code == expected_code
    assert normalized.provider == "deepseek"
    assert normalized.status_code == 401
    assert normalized.request_id == "provider-request-id"
    assert "secret-value" not in str(normalized)
```

- [ ] **Step 2: Run normalization test to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_model_provider.py::test_provider_exceptions_are_normalized_without_secret_messages \
  -v
```

Expected: imports fail because the execution error types do not exist.

- [ ] **Step 3: Implement sanitized execution errors**

Add to `app/providers/openai_compatible.py`:

```python
class ProviderExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        provider: str | None,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"Model provider execution failed: {code}.")
        self.code = code
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id


def normalize_provider_exception(
    exc: Exception,
    provider: str | None,
) -> ProviderExecutionError:
    name = type(exc).__name__
    if name == "AuthenticationError":
        code = "authentication_failed"
    elif name == "RateLimitError":
        code = "rate_limited"
    elif name in {"APITimeoutError", "TimeoutException", "TimeoutError"}:
        code = "timeout"
    elif name in {
        "APIStatusError",
        "APIConnectionError",
        "BadRequestError",
    }:
        code = "protocol_error"
    else:
        code = "provider_execution_failed"
    return ProviderExecutionError(
        code,
        provider=provider,
        status_code=getattr(exc, "status_code", None),
        request_id=getattr(exc, "request_id", None),
    )
```

- [ ] **Step 4: Wrap Runner failures**

In `BusinessAgentExecutor._run_openai_agents`, keep a local `factory` and wrap
only `Runner.run`:

```python
        factory = self._get_model_factory()
        model_adapter = factory.create(model)
        agent = Agent[CustomerServiceState](
            name="Customer Service Agent",
            instructions=instructions,
            model=model_adapter,
            tools=tools,
            mcp_servers=mcp_servers,
        )
        try:
            result = await Runner.run(
                starting_agent=agent,
                input=user_message,
                context=state,
                hooks=hooks,
                max_turns=max_turns,
            )
        except Exception as exc:
            provider = getattr(
                getattr(factory, "config", None),
                "provider",
                None,
            )
            raise normalize_provider_exception(exc, provider) from exc
```

Import `normalize_provider_exception`.

- [ ] **Step 5: Record safe diagnostics in Harness error traces**

In `CustomerServiceHarness.run`, build error detail without messages:

```python
            error_detail = {"error_type": type(exc).__name__}
            if isinstance(
                exc,
                (ProviderConfigurationError, ProviderExecutionError),
            ):
                error_detail.update(
                    {
                        "error_code": exc.code,
                        "provider": exc.provider,
                    }
                )
                if isinstance(exc, ProviderExecutionError):
                    error_detail.update(
                        {
                            "status_code": exc.status_code,
                            "provider_request_id": exc.request_id,
                        }
                    )
            tracer.record(
                "request_error",
                error_detail,
                status="error",
            )
```

Import `ProviderConfigurationError` and `ProviderExecutionError`. Keep the
customer response error exactly `Customer service processing failed.`.

- [ ] **Step 6: Add a Harness redaction regression test**

Append to `tests/test_harness_flow.py`:

```python
@pytest.mark.asyncio
async def test_harness_exposes_only_sanitized_provider_diagnostics():
    class FailingExecutor:
        async def run(self, **kwargs):
            raise ProviderExecutionError(
                "authentication_failed",
                provider="deepseek",
                status_code=401,
                request_id="provider-request-id",
            )

    response = await CustomerServiceHarness(
        executor=FailingExecutor()
    ).run(
        CustomerServiceRequest(
            tenant_id="tenant_a",
            content="Help",
        )
    )

    error_event = next(
        event
        for event in response.state_snapshot["events"]
        if event["name"] == "request_error"
    )
    assert response.error == "Customer service processing failed."
    assert error_event["detail"]["error_code"] == "authentication_failed"
    assert error_event["detail"]["provider"] == "deepseek"
    assert error_event["detail"]["status_code"] == 401
```

Import `ProviderExecutionError` in the test module.

- [ ] **Step 7: Run provider failure tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_model_provider.py \
  tests/test_harness_flow.py::test_harness_exposes_only_sanitized_provider_diagnostics \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Stage 2**

Run:

```bash
git add app/providers/openai_compatible.py \
  app/harness/business_agent_executor.py \
  app/harness/customer_service_harness.py \
  tests/test_model_provider.py \
  tests/test_harness_flow.py
git commit -m "feat: integrate openai-compatible model adapter"
```

- [ ] **Stage 2 Checkpoint: stop and report**

Report adapter construction, executor injection, provider tracing behavior,
tests run, and commit hash.

---

## Stage 3: Deterministic Full-Chain Offline E2E

### Task 3.1: Register E2E Markers and Build Shared Fixtures

**Files:**

- Modify: `pyproject.toml`
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/conftest.py`
- Test: `tests/e2e/test_customer_service_e2e.py`

- [ ] **Step 1: Register pytest markers**

Add to `[tool.pytest.ini_options]`:

```toml
markers = [
  "e2e: deterministic full-chain application tests",
  "live_model: tests that call the configured external model provider",
]
```

- [ ] **Step 2: Write the first failing API-to-Harness E2E test**

Create `tests/e2e/__init__.py` as an empty file.

Create `tests/e2e/test_customer_service_e2e.py`:

```python
import pytest


pytestmark = pytest.mark.e2e


def test_plain_reply_runs_through_api_and_real_harness(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "plain_reply",
            "tenant_id": "tenant_a",
            "content": "Please confirm that support is available.",
            "async_mode": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert "support" in payload["reply"]["body"].lower()
    assert payload["state_snapshot"]["events"][0]["name"] == "request_start"
```

- [ ] **Step 3: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/e2e/test_customer_service_e2e.py::test_plain_reply_runs_through_api_and_real_harness \
  -v
```

Expected: fixture `e2e_client` is not found.

- [ ] **Step 4: Implement the deterministic model boundary**

Create `tests/e2e/conftest.py` with:

```python
import json

import pytest
from agents.tool_context import ToolContext
from fastapi.testclient import TestClient

from app.api.routes import get_harness
from app.harness.business_agent_executor import (
    AgentRunResult,
    BusinessAgentExecutor,
)
from app.harness.customer_service_harness import CustomerServiceHarness
from app.main import app
from app.tools.registry import ToolRegistry


async def invoke_tool(state, tools, name, arguments):
    tool = next(item for item in tools if item.name == name)
    raw = json.dumps(arguments)
    result = await tool.on_invoke_tool(
        ToolContext(
            context=state,
            tool_name=name,
            tool_call_id=f"e2e_{name}",
            tool_arguments=raw,
        ),
        raw,
    )
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


async def deterministic_runner(
    state,
    instructions,
    user_message,
    tools,
    mcp_servers,
    max_turns,
    model,
):
    del instructions, user_message, max_turns, model
    case = state.request_id

    if case == "plain_reply":
        output = "Customer support is available and ready to help."
    elif case == "order_lookup":
        order = await invoke_tool(
            state, tools, "check_order_status", {"order_id": "A100"}
        )
        output = f"Order A100 is {order['status']}."
    elif case == "shipping_lookup":
        shipment = await invoke_tool(
            state, tools, "check_shipping_status", {"order_id": "A100"}
        )
        output = (
            f"Tracking {shipment['tracking_number']} is "
            f"{shipment['status']} with {shipment['carrier']}."
        )
    elif case == "rag_lookup":
        knowledge = await invoke_tool(
            state, tools, "get_rag_knowledge", {"query": "return policy"}
        )
        output = knowledge[0]["text"]
    elif case == "human_handoff":
        await invoke_tool(
            state,
            tools,
            "handoff_to_human",
            {"handoff_type": "reply_handoff", "reason": "customer request"},
        )
        output = "I will connect you with a human support agent."
    elif case == "high_risk":
        output = "A specialist will review this complaint."
    elif case == "german_reply":
        output = (
            "Guten Tag Frau Müller,\n\n"
            "Ihre Bestellung wurde versendet.\n\nKundenservice"
        )
    else:
        output = "Handled by the deterministic E2E runner."

    return AgentRunResult(
        final_output=output,
        token_usage={"total_tokens": 1},
    )


@pytest.fixture
def e2e_client():
    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=deterministic_runner),
        tool_registry=ToolRegistry(),
    )
    app.dependency_overrides[get_harness] = lambda: harness
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 5: Run first E2E test to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  tests/e2e/test_customer_service_e2e.py::test_plain_reply_runs_through_api_and_real_harness \
  -v
```

Expected: PASS.

### Task 3.2: Add Core Offline Business Scenarios

**Files:**

- Modify: `tests/e2e/test_customer_service_e2e.py`
- Modify: `tests/e2e/conftest.py`

- [ ] **Step 1: Add failing order, logistics, RAG, handoff, language, and tenant tests**

Append tests that submit these payloads:

```python
def tool_calls(payload):
    return payload["state_snapshot"]["tool_call_history"]


def test_order_lookup_records_business_tool(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "order_lookup",
            "tenant_id": "tenant_a",
            "content": "Where is order A100?",
            "tools": ["check_order_status"],
        },
    )
    payload = response.json()
    assert tool_calls(payload)[0]["tool_name"] == "check_order_status"
    assert payload["state_snapshot"]["order_result"]["status"] == "shipped"


def test_shipping_lookup_records_tracking_result(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "shipping_lookup",
            "tenant_id": "tenant_a",
            "content": "Track order A100.",
            "tools": ["check_shipping_status"],
        },
    )
    payload = response.json()
    assert tool_calls(payload)[0]["tool_name"] == "check_shipping_status"
    assert "DHL123456" in payload["reply"]["body"]


def test_rag_lookup_honors_kb_and_file_filters(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "rag_lookup",
            "tenant_id": "tenant_a",
            "content": "What is the return policy?",
            "knowledge_config": {
                "knowledges": [
                    {
                        "kbId": "support",
                        "fileIds": ["return_policy"],
                    }
                ],
                "top_k": 1,
                "threshold": 0.1,
            },
        },
    )
    payload = response.json()
    knowledge = payload["state_snapshot"]["retrieved_knowledge"]
    assert [item["file_id"] for item in knowledge] == ["return_policy"]


def test_customer_requested_handoff_runs_full_policy_chain(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "human_handoff",
            "tenant_id": "tenant_a",
            "content": "I want a human agent.",
        },
    )
    payload = response.json()
    assert payload["need_handoff_to_human"] is True
    assert payload["handoff_type"] == "reply_handoff"


def test_high_risk_complaint_suppresses_reply(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "high_risk",
            "tenant_id": "tenant_a",
            "content": "This product injured me and I will take legal action.",
        },
    )
    payload = response.json()
    assert payload["handoff_type"] == "no_reply_handoff"
    assert payload["reply"]["body"] == ""


def test_multilingual_reply_survives_post_processing(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "german_reply",
            "tenant_id": "tenant_a",
            "content": "Wo ist meine Bestellung?",
        },
    )
    body = response.json()["reply"]["body"]
    assert body.startswith("Guten Tag")
    assert body.endswith("Kundenservice")


def test_tenant_b_cannot_read_tenant_a_order(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "order_lookup",
            "tenant_id": "tenant_b",
            "content": "Where is order A100?",
            "tools": ["check_order_status"],
        },
    )
    result = response.json()["state_snapshot"]["order_result"]
    assert result["error"] == "order_not_found"
```

- [ ] **Step 2: Run the E2E module**

Run:

```bash
.venv/bin/python -m pytest tests/e2e/test_customer_service_e2e.py -v
```

Expected: core scenarios pass. Fix only fixture behavior or assertions that
conflict with the approved contracts.

### Task 3.3: Add Offline MCP and Dynamic HTTP Scenarios

**Files:**

- Modify: `tests/e2e/conftest.py`
- Modify: `tests/e2e/test_customer_service_e2e.py`

- [ ] **Step 1: Add deterministic MCP and HTTP runner branches**

Extend `deterministic_runner`:

```python
    elif case == "mcp_manual":
        result = await mcp_servers[0].call_tool(
            "lookup_product_manual",
            {
                "product_or_sku": "Airdog X5",
                "question": "E01 device will not start",
            },
        )
        text = result.structuredContent["result"]
        state.mcp_tool_results["lookup_product_manual"] = {"text": text}
        state.record_tool_call(
            "lookup_product_manual",
            {"product_or_sku": "Airdog X5"},
            {"text": text},
        )
        output = text
    elif case == "dynamic_http":
        result = await invoke_tool(
            state,
            tools,
            "lookup_after_sales_case",
            {"case_id": "RET-42"},
        )
        output = f"After-sales case RET-42 is {result['status']}."
```

Add these imports to `tests/e2e/conftest.py`:

```python
from app.state.request_models import HttpToolConfig, HttpToolParam
```

Add the trusted definition:

```python
TRUSTED_AFTER_SALES_TOOL = HttpToolConfig(
    name="lookup_after_sales_case",
    description="Look up an after-sales case by case id.",
    url="https://mock.local/after-sales",
    method="POST",
    request_params=[
        HttpToolParam(
            name="case_id",
            description="After-sales case id",
            required=True,
        )
    ],
    response_mapping={
        "success_field": "ok",
        "success_value": True,
        "data_field": "payload",
    },
)
```

Add a deterministic business transport fixture that patches only the function
imported by `ToolRegistry`, leaving the model and other HTTP clients untouched:

```python
@pytest.fixture
def e2e_http_client(monkeypatch):
    async def fake_dynamic_http_call(state, config, params):
        assert config == TRUSTED_AFTER_SALES_TOOL
        assert params == {"case_id": "RET-42"}
        result = {
            "case_id": "RET-42",
            "status": "approved",
            "source": "offline_e2e_transport",
        }
        state.http_tool_results[config.name] = result
        state.record_tool_call(config.name, params, result)
        return result

    monkeypatch.setattr(
        "app.tools.registry.call_dynamic_http_tool",
        fake_dynamic_http_call,
    )
    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=deterministic_runner),
        tool_registry=ToolRegistry(
            http_tool_registry={
                "tenant_a": {
                    TRUSTED_AFTER_SALES_TOOL.name:
                        TRUSTED_AFTER_SALES_TOOL,
                }
            }
        ),
    )
    app.dependency_overrides[get_harness] = lambda: harness
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Add the MCP and dynamic HTTP E2E tests**

Add:

```python
def test_local_stdio_mcp_runs_inside_full_chain(e2e_client):
    response = e2e_client.post(
        "/customer-service/respond",
        json={
            "request_id": "mcp_manual",
            "tenant_id": "tenant_a",
            "content": "Airdog X5 shows E01.",
            "mcp_servers": [
                {"name": "product_support", "type": "stdio"}
            ],
        },
    )
    payload = response.json()
    assert "lookup_product_manual" in payload["state_snapshot"][
        "mcp_tool_results"
    ]
    assert "filter compartment" in payload["reply"]["body"]


def test_trusted_dynamic_http_tool_runs_inside_full_chain(e2e_http_client):
    response = e2e_http_client.post(
        "/customer-service/respond",
        json={
            "request_id": "dynamic_http",
            "tenant_id": "tenant_a",
            "content": "Check case RET-42.",
            "http_tools": [
                {
                    "name": "lookup_after_sales_case",
                    "description": "Select this trusted tool.",
                    "url": "https://request-value-is-ignored.example",
                    "method": "GET",
                }
            ],
        },
    )
    payload = response.json()
    call = next(
        item for item in tool_calls(payload)
        if item["tool_name"] == "lookup_after_sales_case"
    )
    assert call["result"]["status"] == "approved"
```

- [ ] **Step 3: Run all offline E2E tests**

Run:

```bash
.venv/bin/python -m pytest -m e2e -v
```

Expected: all offline E2E tests pass without credentials or external network.

- [ ] **Step 4: Run Stage 3 regression**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all existing and new non-live tests pass; live tests do not exist yet.

- [ ] **Step 5: Commit Stage 3**

Run:

```bash
git add pyproject.toml tests/e2e
git commit -m "test: add full-chain offline customer service e2e"
```

- [ ] **Stage 3 Checkpoint: stop and report**

Report scenario coverage, proof that only the model boundary is replaced,
network isolation, test totals, and commit hash.

---

## Stage 4: Opt-In Live Model Validation

### Task 4.1: Add Shared Live-Test Gate

**Files:**

- Modify: `tests/e2e/conftest.py`
- Test: `tests/e2e/test_live_model_smoke.py`

- [ ] **Step 1: Add the live gate fixture**

Add to `tests/e2e/conftest.py`:

```python
import os

from app.config import Settings
from app.providers.openai_compatible import (
    ProviderConfigurationError,
    resolve_provider_config,
)


def live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_MODEL_TESTS", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }


@pytest.fixture
def live_provider_config():
    if not live_tests_enabled():
        pytest.skip("set RUN_LIVE_MODEL_TESTS=1 to enable live model tests")
    try:
        return resolve_provider_config(Settings())
    except ProviderConfigurationError as exc:
        pytest.fail(f"live model configuration error: {exc.code}")
```

- [ ] **Step 2: Write the live smoke test**

Create `tests/e2e/test_live_model_smoke.py`:

```python
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


pytestmark = pytest.mark.live_model


def test_live_provider_runs_customer_service_chain(live_provider_config):
    with TestClient(app) as client:
        response = client.post(
            "/customer-service/respond",
            json={
                "request_id": "live_smoke",
                "tenant_id": "tenant_a",
                "content": (
                    "Reply in English with one short sentence confirming "
                    "that customer support received this message."
                ),
                "max_turns": 2,
            },
        )

    payload = response.json()
    serialized = json.dumps(payload)
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["reply"]["body"].strip()
    assert any(
        event["name"] == "agent_end"
        for event in payload["state_snapshot"]["events"]
    )
    assert live_provider_config.api_key not in serialized
```

- [ ] **Step 3: Verify the default suite skips live tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/e2e/test_live_model_smoke.py \
  -v
```

Expected: one skipped test and zero provider calls.

### Task 4.2: Add Live Core Tool Tests

**Files:**

- Create: `tests/e2e/test_live_model_tools.py`

- [ ] **Step 1: Add independent live tool scenarios**

Create `tests/e2e/test_live_model_tools.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


pytestmark = pytest.mark.live_model


def run_case(payload):
    with TestClient(app) as client:
        response = client.post("/customer-service/respond", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success", result
    return result


def called_tool(result, name):
    return next(
        item
        for item in result["state_snapshot"]["tool_call_history"]
        if item["tool_name"] == name
    )


def test_live_model_calls_order_tool(live_provider_config):
    del live_provider_config
    result = run_case(
        {
            "request_id": "live_order",
            "tenant_id": "tenant_a",
            "content": (
                "Use check_order_status to check order A100, then tell me "
                "the status. Do not guess."
            ),
            "tools": ["check_order_status"],
            "max_turns": 4,
        }
    )
    call = called_tool(result, "check_order_status")
    assert call["arguments"]["order_id"] == "A100"
    assert call["result"]["status"] == "shipped"


def test_live_model_calls_shipping_tool(live_provider_config):
    del live_provider_config
    result = run_case(
        {
            "request_id": "live_shipping",
            "tenant_id": "tenant_a",
            "content": (
                "Use check_shipping_status for order A100 and report the "
                "tracking number. Do not answer before checking."
            ),
            "tools": ["check_shipping_status"],
            "max_turns": 4,
        }
    )
    call = called_tool(result, "check_shipping_status")
    assert call["result"]["tracking_number"] == "DHL123456"
    assert "DHL123456" in result["reply"]["body"]


def test_live_model_calls_filtered_rag_tool(live_provider_config):
    del live_provider_config
    result = run_case(
        {
            "request_id": "live_rag",
            "tenant_id": "tenant_a",
            "content": (
                "Use get_rag_knowledge to find the return policy, then "
                "answer from the retrieved policy only."
            ),
            "knowledge_config": {
                "knowledges": [
                    {
                        "kbId": "support",
                        "fileIds": ["return_policy"],
                    }
                ],
                "top_k": 1,
                "threshold": 0.1,
            },
            "max_turns": 4,
        }
    )
    call = called_tool(result, "get_rag_knowledge")
    assert [item["file_id"] for item in call["result"]] == [
        "return_policy"
    ]


def test_live_model_records_handoff_tool(live_provider_config):
    del live_provider_config
    result = run_case(
        {
            "request_id": "live_handoff",
            "tenant_id": "tenant_a",
            "content": (
                "I explicitly want a human support agent. Call "
                "handoff_to_human before replying."
            ),
            "max_turns": 4,
        }
    )
    called_tool(result, "handoff_to_human")
    assert result["need_handoff_to_human"] is True
```

- [ ] **Step 2: Verify all live tool tests skip by default**

Run:

```bash
.venv/bin/python -m pytest \
  tests/e2e/test_live_model_tools.py \
  -v
```

Expected: all tests skipped and zero provider calls.

### Task 4.3: Add Live MCP and Dynamic HTTP Tests

**Files:**

- Modify: `tests/e2e/test_live_model_tools.py`
- Modify: `tests/e2e/conftest.py`
- Modify: `app/observability/hooks.py`
- Modify: `tests/test_harness_flow.py`

- [ ] **Step 1: Write a failing MCP result recording test**

Append to `tests/test_harness_flow.py`:

```python
@pytest.mark.asyncio
async def test_hooks_record_untracked_sdk_tool_result_as_mcp_result():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Check the manual.",
    )
    hooks = CustomerServiceRunHooks(LocalTracer("req_mcp"))
    context = SimpleNamespace(context=state)
    tool = SimpleNamespace(name="lookup_product_manual")

    await hooks.on_tool_end(
        context,
        SimpleNamespace(name="Customer Service Agent"),
        tool,
        {"text": "manual result"},
    )

    assert state.mcp_tool_results["lookup_product_manual"] == {
        "text": "manual result"
    }
    assert state.tool_call_history[-1] == {
        "tool_name": "lookup_product_manual",
        "arguments": {},
        "result": {"text": "manual result"},
    }
```

- [ ] **Step 2: Run the hook test to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_harness_flow.py::test_hooks_record_untracked_sdk_tool_result_as_mcp_result \
  -v
```

Expected: `mcp_tool_results` remains empty.

- [ ] **Step 3: Record SDK-managed tool results without duplicating function tools**

Update `CustomerServiceRunHooks.on_tool_end` before recording the trace event:

```python
        state = context.context
        tool_name = getattr(tool, "name", "unknown")
        already_recorded = any(
            item.get("tool_name") == tool_name
            for item in state.tool_call_history
        )
        if not already_recorded:
            safe_result = (
                result
                if isinstance(result, (dict, list, str, int, float, bool))
                or result is None
                else {"value": str(result)}
            )
            state.mcp_tool_results[tool_name] = safe_result
            state.record_tool_call(tool_name, {}, safe_result)
```

Keep the existing `tool_end` trace event. Function tools already record their
own arguments and results, so the `already_recorded` check prevents duplicate
history entries.

- [ ] **Step 4: Run focused hook regression**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_harness_flow.py::test_hooks_record_untracked_sdk_tool_result_as_mcp_result \
  tests/test_harness_flow.py::test_customer_service_hooks_record_tool_events \
  -v
```

Expected: both tests pass.

- [ ] **Step 5: Add a trusted local live HTTP fixture**

Add these imports to `tests/e2e/conftest.py`:

```python
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from app.state.request_models import HttpToolConfig, HttpToolParam
```

Add:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_http_client(live_provider_config):
    del live_provider_config
    port = available_port()
    env = os.environ.copy()
    env["MOCK_HTTP_PORT"] = str(port)
    process = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_mock_http_server.py"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{base_url}/docs", timeout=0.2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        process.terminate()
        process.wait(timeout=5)
        pytest.fail("local mock HTTP server did not start")

    trusted = HttpToolConfig(
        name="lookup_after_sales_case",
        description="Look up an after-sales case by order id.",
        url=f"{base_url}/mock/order",
        method="POST",
        request_params=[
            HttpToolParam(
                name="order_id",
                description="After-sales case id represented as an order id",
                required=True,
            )
        ],
    )
    harness = CustomerServiceHarness(
        tool_registry=ToolRegistry(
            http_tool_registry={
                "tenant_a": {trusted.name: trusted}
            }
        )
    )

    def send(payload):
        app.dependency_overrides[get_harness] = lambda: harness
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/customer-service/respond",
                    json=payload,
                )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success", result
            return result
        finally:
            app.dependency_overrides.clear()

    try:
        yield send
    finally:
        process.terminate()
        process.wait(timeout=5)
```

- [ ] **Step 6: Add live MCP and HTTP tool tests**

Add tests with these assertions:

```python
def test_live_model_calls_local_mcp_manual(live_provider_config):
    result = run_case(
        {
            "request_id": "live_mcp",
            "tenant_id": "tenant_a",
            "content": (
                "Use the product_support MCP manual tool for Airdog X5 "
                "error E01 before answering."
            ),
            "mcp_servers": [
                {"name": "product_support", "type": "stdio"}
            ],
            "max_turns": 5,
        }
    )
    called_tool(result, "lookup_product_manual")
    assert "lookup_product_manual" in result["state_snapshot"][
        "mcp_tool_results"
    ]


def test_live_model_calls_trusted_local_http_tool(
    live_provider_config,
    live_http_client,
):
    del live_provider_config
    result = live_http_client(
        {
            "request_id": "live_http",
            "tenant_id": "tenant_a",
            "content": (
                "Use lookup_after_sales_case with order_id RET-42 before "
                "answering."
            ),
            "http_tools": [
                {
                    "name": "lookup_after_sales_case",
                    "description": "Select the trusted after-sales tool.",
                    "url": "https://request-cannot-override.example",
                    "method": "GET",
                }
            ],
            "max_turns": 4,
        }
    )
    call = called_tool(result, "lookup_after_sales_case")
    assert call["arguments"]["order_id"] == "RET-42"
    assert call["result"] == {
        "order_id": "RET-42",
        "status": "shipped",
    }
```

- [ ] **Step 7: Run default live-test selection**

Run:

```bash
.venv/bin/python -m pytest -m live_model -v
```

Expected: all live tests are skipped when `RUN_LIVE_MODEL_TESTS` is false.

- [ ] **Step 8: Run a configured live smoke test**

After the user fills the active provider's API key and model in `.env`, run:

```bash
RUN_LIVE_MODEL_TESTS=1 \
  .venv/bin/python -m pytest \
  tests/e2e/test_live_model_smoke.py \
  -v
```

Expected: PASS for the selected provider. If credentials have not been filled,
report the configuration error and do not claim live validation passed.

- [ ] **Step 9: Run configured live core tool tests**

Run:

```bash
RUN_LIVE_MODEL_TESTS=1 \
  .venv/bin/python -m pytest \
  tests/e2e/test_live_model_tools.py \
  -v
```

Expected: selected provider calls the required local tools. Record provider,
model, pass/fail, token usage, and latency without recording the API key.

- [ ] **Step 10: Commit Stage 4**

Run:

```bash
git add tests/e2e/test_live_model_smoke.py \
  tests/e2e/test_live_model_tools.py \
  tests/e2e/conftest.py \
  app/observability/hooks.py \
  tests/test_harness_flow.py
git commit -m "test: add opt-in live model validation"
```

- [ ] **Stage 4 Checkpoint: stop and report**

Report default skip proof, selected provider configuration status, live smoke
result, each tool scenario result, costs or token usage when available, and
commit hash. Do not report live success if credentials were not configured.

---

## Stage 5: Documentation, Security Audit, and Final Verification

### Task 5.1: Document Provider Setup and Test Workflows

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update README environment setup**

Document:

```bash
cp .env.example .env
```

Explain that all three providers can be filled in, while `MODEL_PROVIDER`
selects exactly one active provider for the process. Include these examples:

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
ALLOWED_MODELS=...
```

```dotenv
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
ALLOWED_MODELS=deepseek-chat
```

```dotenv
MODEL_PROVIDER=dashscope
DASHSCOPE_API_KEY=...
DASHSCOPE_MODEL=...
ALLOWED_MODELS=...
```

State that provider model names depend on the user's account and region and
must be filled with a model actually available to that account.

- [ ] **Step 2: Document offline and live commands**

Add:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m pytest -m e2e -v
.venv/bin/python -m pytest -m live_model -v
RUN_LIVE_MODEL_TESTS=1 .venv/bin/python -m pytest -m live_model -v
```

Explain that live tests consume quota and use real model calls but only local
mock business systems.

- [ ] **Step 3: Update AGENTS.md**

Add Phase 6 conventions:

- provider resolution belongs in `app/providers/`
- business code must not branch on provider name
- `.env` must never be committed
- live tests require `RUN_LIVE_MODEL_TESTS=1`
- default tests must remain network-independent
- live failures require reporting provider/model and sanitized error class

### Task 5.2: Run Security and Completion Verification

**Files:**

- Modify: `docs/superpowers/plans/2026-06-20-phase6-live-model-and-e2e-validation-implementation-plan.md`

- [ ] **Step 1: Check secrets and ignored files**

Run:

```bash
git check-ignore .env
git ls-files .env
git grep -nE '(sk-[A-Za-z0-9_-]{16,}|Bearer [A-Za-z0-9._-]{16,})' \
  -- ':!.env'
```

Expected:

- `.env` is ignored.
- `git ls-files .env` prints nothing.
- secret scan prints no real credential.

- [ ] **Step 2: Run configuration and adapter tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_model_provider.py \
  tests/test_harness_flow.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 3: Run deterministic E2E tests**

Run:

```bash
.venv/bin/python -m pytest -m e2e -v
```

Expected: all offline full-chain tests pass without credentials.

- [ ] **Step 4: Verify live tests skip by default**

Run:

```bash
RUN_LIVE_MODEL_TESTS=0 \
  .venv/bin/python -m pytest -m live_model -v
```

Expected: all live tests skipped.

- [ ] **Step 5: Run the complete non-live suite**

Run:

```bash
RUN_LIVE_MODEL_TESTS=0 .venv/bin/python -m pytest -q
```

Expected: zero failures. Record the exact pass and skip totals.

- [ ] **Step 6: Optionally run selected-provider live acceptance**

Only when `.env` contains the selected provider's real key and model:

```bash
RUN_LIVE_MODEL_TESTS=1 \
  .venv/bin/python -m pytest -m live_model -v
```

Expected: all selected-provider live tests pass. If credentials are absent,
mark live acceptance as pending user configuration rather than failed.

- [ ] **Step 7: Update plan status**

Mark every completed task and stage in this document. Record any live tests
that remain pending because credentials were not supplied.

- [ ] **Step 8: Commit Stage 5**

Run:

```bash
git add README.md AGENTS.md \
  docs/superpowers/plans/2026-06-20-phase6-live-model-and-e2e-validation-implementation-plan.md
git commit -m "docs: complete phase 6 validation workflow"
```

- [ ] **Stage 5 Checkpoint: stop and report**

Report changed files, all commits, exact test totals, live provider result or
pending status, security scan result, remaining limitations, and recommended
branch-finishing action.
