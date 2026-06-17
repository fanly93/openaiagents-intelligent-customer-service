from app.state.request_models import (
    CustomerProfile,
    CustomerServiceRequest,
    SlotDefinition,
)


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
