# OpenAI Agents Intelligent Customer Service

## Specification Authority

- The approved project design is:
  `docs/superpowers/specs/2026-06-17-openai-agents-customer-service-design.md`.
- The active implementation plan is:
  `docs/superpowers/plans/2026-06-17-openai-agents-customer-service-demo-implementation-plan.md`.
- Treat those documents as the current implementation baseline.
- Parent-directory architecture notes are exploratory recommendations. They do not
  replace the approved project spec or plan unless the user explicitly approves a
  new design through the Superpowers brainstorming and planning workflow.
- When the spec, plan, and implementation differ, stop and reconcile the difference
  before broad architectural changes.

## Current Architecture Baseline

- Build the first version as a near-production customer service backend demo.
- Use `tenant_id` as the tenant identifier defined by the approved spec.
- Use one OpenAI Agents SDK customer service Agent coordinated by a lightweight
  `CustomerServiceHarness`.
- Keep lifecycle responsibilities separated across prompt assembly, tool and MCP
  preparation, Agent execution, reply post-processing, handoff policy, state
  reduction, retrieval, and observability modules.
- Use request-driven runtime configuration for tools, MCP servers, knowledge,
  memory, model, and instructions.
- Keep mock data behind provider boundaries so production services can replace
  providers without rewriting the core harness.
- Do not introduce Router, TaskPlanner, lazy skills, or specialist-agent
  orchestration in the first version unless a revised spec is explicitly approved.
- Tools may update the shared `CustomerServiceState` through the OpenAI Agents SDK
  context in this first version. `StateReducer` is responsible for producing the
  final response and enforcing response-level consistency.

## Development Workflow

- Work on `feature/customer-service-agent-demo`, not directly on `main`.
- Follow the approved five-phase plan and stop for user review after each phase.
- Use test-driven development for behavior changes: add a focused failing test,
  verify the expected failure, implement the minimum behavior, then verify the
  focused and regression test suites.
- Assess dependencies and shared files before dispatching subagents. Do not
  parallelize tasks that have sequential dependencies or modify the same files.
- Commit each completed task or tightly related task group.

## Development Environment

- Project root:
  `/Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service`
- Python requirement: Python 3.11 or later.
- Current local development version: Python 3.13.12.
- The project-local virtual environment is `.venv`.
- Do not install project dependencies into the system Python, the user Python, or the Miniconda base environment.
- Run Python, pip, pytest, uvicorn, and project scripts through `.venv`.

Create and initialize the environment when `.venv` does not exist:

```bash
cd /Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service
/opt/miniconda3/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

For interactive development, activate the environment first:

```bash
source .venv/bin/activate
```

For automation and agent-executed commands, prefer explicit executable paths so the
environment does not depend on shell activation:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m pip install -e ".[dev]"
```

Before installing packages or running tests, verify the interpreter:

```bash
.venv/bin/python -c "import sys; print(sys.executable)"
```

The printed path must be inside:

`/Users/tanglin/VibeCoding/intelligent-customer-service/openaiagents-intelligent-customer-service/.venv`

The `.venv/` directory is local-only and must remain ignored by Git.
