from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    name: str
    status: Literal["started", "success", "error"] = "success"
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LocalTracer:
    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        detail: dict[str, Any] | None = None,
        status: Literal["started", "success", "error"] = "success",
    ) -> None:
        self.events.append(TraceEvent(name=name, status=status, detail=detail or {}))


class OptionalLangfuseTracer:
    def __init__(self) -> None:
        self.enabled = False
        self.client = None

    def try_start(self) -> bool:
        try:
            from app.config import get_settings

            settings = get_settings()
            if not (
                settings.langfuse_public_key
                and settings.langfuse_secret_key
            ):
                return False
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            self.client = None
            self.enabled = False
            return False

        self.enabled = True
        return True
