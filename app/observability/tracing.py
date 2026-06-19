from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    name: str
    status: Literal["started", "success", "error"] = "success"
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LocalTracer:
    def __init__(
        self,
        request_id: str | None = None,
        external_tracer: Any | None = None,
    ) -> None:
        self.request_id = request_id
        self.events: list[TraceEvent] = []
        self.external_tracer = external_tracer

    def record(
        self,
        name: str,
        detail: dict[str, Any] | None = None,
        status: Literal["started", "success", "error"] = "success",
    ) -> None:
        payload = detail or {}
        self.events.append(TraceEvent(name=name, status=status, detail=payload))
        if self.external_tracer:
            self.external_tracer.record_event(name, payload, status)


class OptionalLangfuseTracer:
    def __init__(self, client_factory: Any | None = None) -> None:
        self.enabled = False
        self.client = None
        self.trace = None
        self.client_factory = client_factory

    def try_start(self, request_id: str | None = None) -> bool:
        try:
            from app.config import get_settings

            settings = get_settings()
            if not (
                settings.langfuse_public_key
                and settings.langfuse_secret_key
            ):
                return False
            if self.client_factory is None:
                from langfuse import Langfuse

                self.client_factory = Langfuse
            self.client = self.client_factory(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self.trace = self.client.trace(
                name="customer_service_request",
                id=request_id,
            )
        except Exception:
            self.client = None
            self.trace = None
            self.enabled = False
            return False

        self.enabled = True
        return True

    def record_event(
        self,
        name: str,
        detail: dict[str, Any],
        status: str,
    ) -> None:
        if not self.enabled or not self.trace:
            return
        try:
            self.trace.event(
                name=name,
                metadata={"status": status, **detail},
            )
        except Exception:
            self.enabled = False

    def flush(self) -> None:
        if not self.client:
            return
        try:
            self.client.flush()
        except Exception:
            self.enabled = False
