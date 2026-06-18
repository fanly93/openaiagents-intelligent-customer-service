from typing import Any

from app.state.conversation_state import CustomerServiceState
from app.state.result_models import CustomerServiceResponse


_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "email",
    "password",
    "secret",
    "token",
}


def _redact_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.casefold() in _SENSITIVE_KEYS
            else _redact_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_snapshot(item) for item in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "...[TRUNCATED]"
    return value


class StateReducer:
    def build_response(
        self,
        state: CustomerServiceState,
        processing_time: float,
        error: str | None = None,
    ) -> CustomerServiceResponse:
        body = "" if state.handoff_type == "no_reply_handoff" else (state.final_reply or "")
        return CustomerServiceResponse(
            request_id=state.request_id,
            status="error" if error else "success",
            reply={
                "subject": state.subject or "Customer support",
                "body": body,
            },
            need_handoff_to_human=state.need_handoff_to_human,
            handoff_type=state.handoff_type,
            handoff_reason=state.handoff_reason,
            token_usage=state.token_usage,
            processing_time=processing_time,
            error=error,
            state_snapshot=_redact_snapshot(state.model_dump(mode="json")),
        )
