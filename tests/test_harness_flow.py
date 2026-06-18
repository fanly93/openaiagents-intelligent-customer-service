from app.observability.perf import PerformanceTracker
from app.observability.tracing import LocalTracer


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
