from app.state.request_models import (
    CustomerProfile,
    CustomerServiceRequest,
    SlotDefinition,
)
from app.state.conversation_state import CustomerServiceState, SlotValue
from app.state.result_models import CustomerServiceResponse


def test_customer_service_request_defaults_to_email_channel():
    req = CustomerServiceRequest(
        tenant_id="tenant_a",
        content="Where is my order #A100?",
    )

    assert req.channel == "email"
    assert req.async_mode is False
    assert req.max_turns == 6
    assert req.customer == CustomerProfile()


def test_slot_definition_keeps_aliases_and_required_flag():
    slot = SlotDefinition(
        name="product_model",
        description="Product model",
        required=True,
        aliases=["model", "型号"],
    )

    assert slot.name == "product_model"
    assert slot.required is True
    assert slot.aliases == ["model", "型号"]


def test_state_tracks_collected_slots_and_missing_required_slots():
    state = CustomerServiceState(
        request_id="req_1",
        tenant_id="tenant_a",
        channel="email",
        content="My Airdog X5 will not turn on",
        slot_schema=[
            SlotDefinition(name="product_model", description="Product model", required=True),
            SlotDefinition(name="symptom", description="Problem symptom", required=True),
        ],
    )

    state.collected_slots["product_model"] = SlotValue(
        value="Airdog X5",
        confidence=0.9,
        source="current_message",
    )
    state.refresh_missing_required_slots()

    assert state.missing_required_slots == ["symptom"]


def test_response_contains_handoff_fields():
    response = CustomerServiceResponse(
        request_id="req_1",
        status="success",
        reply={"subject": "Re: Support", "body": "We can help."},
        need_handoff_to_human=False,
        handoff_type="no_handoff",
    )

    assert response.reply["body"] == "We can help."
    assert response.need_handoff_to_human is False
