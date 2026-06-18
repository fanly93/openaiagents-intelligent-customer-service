from typing import Any

from app.state.conversation_state import CustomerServiceState, SlotValue


async def extract_slots_impl(
    state: CustomerServiceState,
    slots: list[dict[str, Any]],
) -> dict[str, list[str]]:
    schema_names = {slot.name for slot in state.slot_schema}
    accepted: list[str] = []
    ignored: list[str] = []

    for item in slots:
        name = str(item.get("name", "")).strip()
        if name not in schema_names:
            ignored.append(name)
            continue

        state.collected_slots[name] = SlotValue(
            value=item.get("value"),
            confidence=float(item.get("confidence", 1.0)),
            source=item.get("source", "current_message"),
        )
        accepted.append(name)

    result = {"accepted": accepted, "ignored": ignored}
    state.refresh_missing_required_slots()
    state.record_tool_call("extract_slots", {"slots": slots}, result)
    return result
