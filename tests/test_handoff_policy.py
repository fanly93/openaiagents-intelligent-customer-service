import pytest

from app.harness.handoff_policy import HandoffPolicy
from app.state.conversation_state import CustomerServiceState


def make_state(
    *,
    content: str = "Help",
    final_reply: str | None = "We are reviewing your request.",
) -> CustomerServiceState:
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content=content,
    )
    state.final_reply = final_reply
    return state


@pytest.mark.parametrize("final_reply", ["", "Short"])
def test_handoff_policy_marks_empty_or_short_reply_for_reply_handoff(
    final_reply: str,
):
    state = make_state(final_reply=final_reply)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "reply_handoff"
    assert state.handoff_reason == "reply is empty or too short"


def test_handoff_policy_marks_high_risk_request_for_no_reply_handoff():
    state = make_state(content="I will file a lawsuit over this order.")

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "no_reply_handoff"
    assert state.handoff_reason == "high risk complaint or escalation"


@pytest.mark.parametrize(
    "content",
    [
        "I am considering legal action.",
        "I am making a legal complaint about this order.",
        "My lawyer will contact you.",
        "I will file a lawsuit over this order.",
    ],
)
def test_handoff_policy_marks_legal_escalation_for_no_reply_handoff(content: str):
    state = make_state(content=content)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "no_reply_handoff"
    assert state.handoff_reason == "high risk complaint or escalation"


@pytest.mark.parametrize(
    "content",
    [
        "Please update my legal name.",
        "What is the legal entity name on the invoice?",
    ],
)
def test_handoff_policy_ignores_benign_legal_name_language(content: str):
    state = make_state(content=content)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is False
    assert state.handoff_type == "no_handoff"
    assert state.handoff_reason is None


@pytest.mark.parametrize(
    "content",
    [
        "This product is unsafe to use.",
        "The defective charger caused an injury.",
        "这个产品导致我受伤了。",
        "I demand large compensation for this incident.",
        "我要求高额赔偿。",
    ],
)
def test_handoff_policy_marks_safety_or_high_compensation_request_for_no_reply_handoff(
    content: str,
):
    state = make_state(content=content)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "no_reply_handoff"
    assert state.handoff_reason == "high risk complaint or escalation"


@pytest.mark.parametrize(
    "content",
    [
        "Please connect me with a human agent.",
        "I need to speak to a representative.",
        "I need a human.",
        "I want an agent.",
        "Please let me speak with an agent.",
        "Transfer me to a representative.",
        "请帮我转人工客服。",
    ],
)
def test_handoff_policy_marks_explicit_human_request_for_reply_handoff(
    content: str,
):
    state = make_state(content=content)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "reply_handoff"
    assert state.handoff_reason == "customer requested human support"


@pytest.mark.parametrize(
    "content",
    [
        "This is a representative sample of recent orders.",
        "The cleaning agent removed the stain.",
    ],
)
def test_handoff_policy_ignores_non_human_agent_language(content: str):
    state = make_state(content=content)

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is False
    assert state.handoff_type == "no_handoff"
    assert state.handoff_reason is None


def test_handoff_policy_corrects_missing_handoff_type():
    state = make_state()
    state.need_handoff_to_human = True
    state.handoff_type = "no_handoff"

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "reply_handoff"
    assert state.handoff_reason == "corrected inconsistent handoff state"


def test_handoff_policy_clears_handoff_type_when_handoff_is_not_needed():
    state = make_state()
    state.need_handoff_to_human = False
    state.handoff_type = "reply_handoff"
    state.handoff_reason = "stale handoff"

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is False
    assert state.handoff_type == "no_handoff"
    assert state.handoff_reason is None


def test_handoff_policy_preserves_existing_no_reply_handoff():
    state = make_state(
        content="Please connect me with a human.",
        final_reply="Short",
    )
    state.need_handoff_to_human = True
    state.handoff_type = "no_reply_handoff"
    state.handoff_reason = "payment fraud review"

    HandoffPolicy().apply(state)

    assert state.need_handoff_to_human is True
    assert state.handoff_type == "no_reply_handoff"
    assert state.handoff_reason == "payment fraud review"
