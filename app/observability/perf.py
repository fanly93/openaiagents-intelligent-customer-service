from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class PerformanceTracker:
    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self._records.setdefault(name, []).append(elapsed)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "total_seconds": round(sum(values), 6),
                "max_seconds": round(max(values), 6),
            }
            for name, values in self._records.items()
        }
