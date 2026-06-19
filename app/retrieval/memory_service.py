import json
import re
from pathlib import Path
from typing import Any


class MockMemoryService:
    def __init__(self, data_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.data_path = (
            Path(data_path)
            if data_path is not None
            else project_root / "data/mock_memories.json"
        )

    async def retrieve(
        self,
        tenant_id: str,
        customer_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not tenant_id or not customer_id or top_k <= 0:
            return []

        data = self._load_data()
        tenant_memories = data.get(tenant_id, {})
        if not isinstance(tenant_memories, dict):
            return []
        memories = tenant_memories.get(customer_id, [])
        if not isinstance(memories, list):
            return []

        query_terms = self._terms(query)
        if not query_terms:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            keywords = memory.get("keywords", [])
            if not isinstance(keywords, list):
                continue
            keyword_terms = {
                term
                for keyword in keywords
                for term in self._terms(str(keyword))
            }
            score = len(query_terms & keyword_terms)
            if score > 0:
                result = dict(memory)
                result["score"] = score
                scored.append((score, result))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _load_data(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _terms(value: str) -> set[str]:
        return set(re.findall(r"\w+", value.lower()))
