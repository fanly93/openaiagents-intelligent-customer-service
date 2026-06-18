import pytest

from app.state.conversation_state import CustomerServiceState
from app.state.request_models import SlotDefinition
from app.tools.core_tools import extract_slots_impl


@pytest.mark.asyncio
async def test_extract_slots_writes_configured_slots_only():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        channel="email",
        content="Airdog X5 has E01",
        slot_schema=[
            SlotDefinition(
                name="product_model",
                description="Product model",
                required=True,
            ),
            SlotDefinition(
                name="error_code",
                description="Error code",
                required=False,
            ),
        ],
    )
    slots = [
        {
            "name": "product_model",
            "value": "Airdog X5",
            "confidence": 0.9,
            "source": "current_message",
        },
        {
            "name": "unknown",
            "value": "ignored",
            "confidence": 0.9,
            "source": "current_message",
        },
    ]

    result = await extract_slots_impl(state, slots)

    assert result == {"accepted": ["product_model"], "ignored": ["unknown"]}
    assert state.collected_slots["product_model"].value == "Airdog X5"
    assert "unknown" not in state.collected_slots
    assert state.tool_call_history == [
        {
            "tool_name": "extract_slots",
            "arguments": {"slots": slots},
            "result": result,
        }
    ]


@pytest.mark.asyncio
async def test_extract_slots_refreshes_missing_required_slots():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Airdog X5",
        slot_schema=[
            SlotDefinition(
                name="product_model",
                description="Product model",
                required=True,
            ),
            SlotDefinition(
                name="symptom",
                description="Problem symptom",
                required=True,
            ),
        ],
        missing_required_slots=["stale_slot"],
    )

    await extract_slots_impl(
        state,
        [{"name": "product_model", "value": "Airdog X5"}],
    )

    assert state.missing_required_slots == ["symptom"]


@pytest.mark.asyncio
async def test_extract_slots_preserves_valid_source_and_confidence():
    state = CustomerServiceState(
        tenant_id="tenant_a",
        content="Earlier I mentioned error E01",
        slot_schema=[
            SlotDefinition(name="error_code", description="Error code"),
        ],
    )

    await extract_slots_impl(
        state,
        [
            {
                "name": "error_code",
                "value": "E01",
                "confidence": 0.73,
                "source": "history",
            }
        ],
    )

    slot_value = state.collected_slots["error_code"]
    assert slot_value.confidence == 0.73
    assert slot_value.source == "history"
