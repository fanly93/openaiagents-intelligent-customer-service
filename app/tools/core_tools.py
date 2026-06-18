import math
from typing import Any

from app.state.conversation_state import CustomerServiceState, SlotValue
from app.state.request_models import SlotDefinition


_ALLOWED_SOURCES = {"current_message", "history", "tool", "memory"}


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().casefold()


def _is_valid_value(slot: SlotDefinition, value: Any) -> bool:
    if value is None or value == "":
        return False
    if slot.type == "string":
        return isinstance(value, str)
    if slot.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if slot.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if slot.type == "boolean":
        return isinstance(value, bool)
    if slot.type == "list":
        return isinstance(value, list)
    return False


def _parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


async def extract_slots_impl(
    state: CustomerServiceState,
    slots: list[dict[str, Any]],
) -> dict[str, list[str]]:
    aliases: dict[str, SlotDefinition] = {}
    for slot in state.slot_schema:
        aliases[_normalize_name(slot.name)] = slot
        for alias in slot.aliases:
            aliases.setdefault(_normalize_name(alias), slot)

    candidates: dict[str, SlotValue] = {}
    ignored: list[str] = []
    invalid: list[str] = []

    for item in slots:
        raw_name = str(item.get("name", "")).strip()
        slot = aliases.get(_normalize_name(raw_name))
        if slot is None:
            ignored.append(raw_name)
            continue

        value = item.get("value")
        confidence = _parse_confidence(item.get("confidence", 1.0))
        source = item.get("source", "current_message")
        if (
            not _is_valid_value(slot, value)
            or confidence is None
            or source not in _ALLOWED_SOURCES
        ):
            invalid.append(slot.name)
            continue

        candidates[slot.name] = SlotValue(
            value=value,
            confidence=confidence,
            source=source,
        )

    accepted = [] if invalid else list(candidates)
    if not invalid:
        state.collected_slots.update(candidates)

    result = {
        "accepted": accepted,
        "ignored": ignored,
    }
    if invalid:
        result["invalid"] = invalid
    state.refresh_missing_required_slots()
    state.record_tool_call("extract_slots", {"slots": slots}, result)
    return result
