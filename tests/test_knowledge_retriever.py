import json

import pytest

from app.retrieval.knowledge_retriever import MockKnowledgeRetriever
from app.state.request_models import KnowledgeConfig, KnowledgeItemConfig


def _write_docs(tmp_path, docs):
    data_dir = tmp_path / "knowledge"
    data_dir.mkdir()
    (data_dir / "docs.json").write_text(json.dumps(docs), encoding="utf-8")
    return data_dir


@pytest.mark.asyncio
async def test_mock_knowledge_retriever_reads_default_data_and_filters_file_ids():
    result = await MockKnowledgeRetriever().retrieve(
        tenant_id="tenant_a",
        query="return refund policy",
        knowledge_config=KnowledgeConfig(
            knowledges=[
                KnowledgeItemConfig(
                    kbId="kb_after_sales",
                    fileIds=["return_policy"],
                )
            ],
            top_k=5,
            threshold=0.1,
        ),
    )

    assert result
    assert all(item["file_id"] == "return_policy" for item in result)
    assert all(0.0 <= item["score"] <= 1.0 for item in result)


@pytest.mark.asyncio
async def test_mock_knowledge_retriever_filters_by_tenant(tmp_path):
    data_dir = _write_docs(
        tmp_path,
        [
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_support",
                "file_id": "tenant_a_manual",
                "text": "Restart the device to clear error E01.",
                "keywords": ["restart", "E01"],
            },
            {
                "tenant_id": "tenant_b",
                "kb_id": "kb_support",
                "file_id": "tenant_b_manual",
                "text": "Restart the device to clear error E01.",
                "keywords": ["restart", "E01"],
            },
        ],
    )

    result = await MockKnowledgeRetriever(data_dir=data_dir).retrieve(
        tenant_id="tenant_a",
        query="restart E01",
    )

    assert [item["file_id"] for item in result] == ["tenant_a_manual"]


@pytest.mark.asyncio
async def test_is_all_ignores_file_ids_for_its_knowledge_item(tmp_path):
    data_dir = _write_docs(
        tmp_path,
        [
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_after_sales",
                "file_id": "return_policy",
                "text": "Return policy allows a refund.",
                "keywords": ["return", "refund"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_after_sales",
                "file_id": "warranty_policy",
                "text": "Warranty policy covers manufacturing defects.",
                "keywords": ["warranty", "defects"],
            },
        ],
    )

    result = await MockKnowledgeRetriever(data_dir=data_dir).retrieve(
        tenant_id="tenant_a",
        query="policy",
        knowledge_config=KnowledgeConfig(
            knowledges=[
                KnowledgeItemConfig(
                    kbId="kb_after_sales",
                    fileIds=["return_policy"],
                    isAll=True,
                )
            ],
            threshold=0.1,
        ),
    )

    assert {item["file_id"] for item in result} == {
        "return_policy",
        "warranty_policy",
    }


@pytest.mark.asyncio
async def test_multiple_knowledge_scopes_keep_file_ids_bound_to_their_kb(tmp_path):
    data_dir = _write_docs(
        tmp_path,
        [
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_a",
                "file_id": "file_a",
                "text": "Shared setup guide alpha.",
                "keywords": ["shared", "setup"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_a",
                "file_id": "file_b",
                "text": "Shared setup guide beta.",
                "keywords": ["shared", "setup"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_b",
                "file_id": "file_b",
                "text": "Shared setup guide gamma.",
                "keywords": ["shared", "setup"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_b",
                "file_id": "file_a",
                "text": "Shared setup guide delta.",
                "keywords": ["shared", "setup"],
            },
        ],
    )

    result = await MockKnowledgeRetriever(data_dir=data_dir).retrieve(
        tenant_id="tenant_a",
        query="shared setup",
        knowledge_config=KnowledgeConfig(
            knowledges=[
                KnowledgeItemConfig(kbId="kb_a", fileIds=["file_a"]),
                KnowledgeItemConfig(kbId="kb_b", fileIds=["file_b"]),
            ],
            threshold=0.1,
        ),
    )

    assert {(item["kb_id"], item["file_id"]) for item in result} == {
        ("kb_a", "file_a"),
        ("kb_b", "file_b"),
    }


@pytest.mark.asyncio
async def test_kb_id_list_top_k_and_threshold_are_applied(tmp_path):
    data_dir = _write_docs(
        tmp_path,
        [
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_a",
                "file_id": "exact",
                "text": "Alpha beta troubleshooting.",
                "keywords": ["alpha", "beta"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_b",
                "file_id": "partial",
                "text": "Alpha troubleshooting.",
                "keywords": ["alpha"],
            },
            {
                "tenant_id": "tenant_a",
                "kb_id": "kb_c",
                "file_id": "outside_scope",
                "text": "Alpha beta troubleshooting.",
                "keywords": ["alpha", "beta"],
            },
        ],
    )

    result = await MockKnowledgeRetriever(data_dir=data_dir).retrieve(
        tenant_id="tenant_a",
        query="alpha beta",
        knowledge_config=KnowledgeConfig(
            knowledges=[
                KnowledgeItemConfig(kbId=["kb_a", "kb_b"], isAll=True)
            ],
            top_k=1,
            threshold=0.75,
        ),
    )

    assert [item["file_id"] for item in result] == ["exact"]
    assert result[0]["score"] == 1.0
