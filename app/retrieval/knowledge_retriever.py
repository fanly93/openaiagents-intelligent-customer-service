import json
import re
from pathlib import Path
from typing import Any

from app.state.request_models import KnowledgeConfig, KnowledgeItemConfig


class MockKnowledgeRetriever:
    def __init__(self, data_dir: str | Path = "data/knowledge") -> None:
        self.data_dir = Path(data_dir)

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        knowledge_config: KnowledgeConfig | None = None,
    ) -> list[dict[str, Any]]:
        top_k = knowledge_config.top_k if knowledge_config else 5
        threshold = knowledge_config.threshold if knowledge_config else 0.0
        query_terms = self._terms(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._load_docs():
            if doc.get("tenant_id") != tenant_id:
                continue
            if not self._matches_knowledge_config(doc, knowledge_config):
                continue

            searchable_terms = self._terms(doc.get("text", ""))
            searchable_terms.update(
                term
                for keyword in doc.get("keywords", [])
                for term in self._terms(str(keyword))
            )
            score = len(query_terms & searchable_terms) / max(len(query_terms), 1)
            if score >= threshold:
                result = dict(doc)
                result["score"] = round(score, 4)
                scored.append((score, result))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _load_docs(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("*.json")):
            loaded = json.loads(path.read_text(encoding="utf-8"))
            docs.extend(loaded)
        return docs

    @classmethod
    def _matches_knowledge_config(
        cls,
        doc: dict[str, Any],
        config: KnowledgeConfig | None,
    ) -> bool:
        if config is None or not config.knowledges:
            return True
        return any(cls._matches_knowledge_item(doc, item) for item in config.knowledges)

    @staticmethod
    def _matches_knowledge_item(
        doc: dict[str, Any],
        item: KnowledgeItemConfig,
    ) -> bool:
        if isinstance(item.kbId, str):
            kb_ids = {item.kbId}
        elif isinstance(item.kbId, list):
            kb_ids = set(item.kbId)
        else:
            kb_ids = set()

        if kb_ids and doc.get("kb_id") not in kb_ids:
            return False
        if not item.isAll and item.fileIds and doc.get("file_id") not in item.fileIds:
            return False
        return True

    @staticmethod
    def _terms(value: str) -> set[str]:
        return set(re.findall(r"\w+", value.lower()))
