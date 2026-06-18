from app.harness.prompt_assembler import PromptAssembler
from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer
from app.state.conversation_state import CustomerServiceState
from app.state.request_models import CustomerProfile, SlotDefinition


def test_performance_tracker_records_named_operation():
    tracker = PerformanceTracker()
    with tracker.track("prepare_tools"):
        pass

    summary = tracker.summary()
    assert "prepare_tools" in summary
    assert summary["prepare_tools"]["count"] == 1


def test_local_tracer_records_events():
    tracer = LocalTracer(request_id="req_1")
    tracer.record("agent_start", {"model": "gpt-4.1-mini"})

    assert tracer.events[0].name == "agent_start"
    assert tracer.events[0].detail["model"] == "gpt-4.1-mini"


def test_prompt_assembler_includes_runtime_context_and_language_rules():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="My device has E01",
        customer=CustomerProfile(name="Ada"),
        slot_schema=[SlotDefinition(name="error_code", description="Device error code")],
    )

    instructions = PromptAssembler().build_instructions(
        state,
        tool_guide="tools here",
        extra_instructions="Follow tenant refund policy.",
    )

    assert "error_code" in instructions
    assert "Use the customer's language" in instructions
    assert "tools here" in instructions
    assert "Follow tenant refund policy." in instructions
