import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
DEMO_SCRIPT = PROJECT_ROOT / "scripts/run_demo_cases.py"
EXPECTED_CASES = {
    "english_order_lookup",
    "shipping_delay",
    "return_policy",
    "electronics_troubleshooting",
    "pet_product_advice",
    "wig_recommendation",
    "human_requested",
    "high_risk_complaint",
    "german_reply",
    "dynamic_http_tool",
}


def _tool_calls(response):
    return response["state_snapshot"]["tool_call_history"]


def _tool_call(response, tool_name):
    return next(
        call for call in _tool_calls(response)
        if call["tool_name"] == tool_name
    )


def _slot_values(response):
    return {
        name: slot["value"]
        for name, slot in response["state_snapshot"]["collected_slots"].items()
    }


def test_demo_script_runs_offline_from_non_project_cwd(tmp_path):
    network_guard_marker = tmp_path / "network-guard-loaded"
    (tmp_path / "sitecustomize.py").write_text(
        """
import os
from pathlib import Path
import socket

Path(os.environ["DEMO_NETWORK_GUARD_MARKER"]).write_text(
    "loaded",
    encoding="utf-8",
)

def block_network(*args, **kwargs):
    raise RuntimeError("network access is disabled for the demo test")

socket.create_connection = block_network
socket.socket.connect = block_network
socket.socket.connect_ex = block_network
""".strip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["DEMO_NETWORK_GUARD_MARKER"] = str(network_guard_marker)
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_ORG_ID", None)
    env.pop("OPENAI_PROJECT_ID", None)

    result = subprocess.run(
        [str(PYTHON), str(DEMO_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert network_guard_marker.read_text(encoding="utf-8") == "loaded"
    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    output = [json.loads(line) for line in lines]

    assert len(output) == 10
    assert all(set(item) == {"case_name", "response"} for item in output)
    assert {item["case_name"] for item in output} == EXPECTED_CASES
    assert len({item["case_name"] for item in output}) == 10
    by_case = {
        item["case_name"]: item["response"]
        for item in output
    }

    assert all(response["status"] == "success" for response in by_case.values())
    assert all(response["error"] is None for response in by_case.values())

    order = by_case["english_order_lookup"]
    assert _slot_values(order)["order_id"] == "A100"
    assert _tool_call(order, "check_order_status")["result"]["status"] == "shipped"
    assert "shipped" in order["reply"]["body"].lower()

    shipping = by_case["shipping_delay"]
    assert _tool_call(shipping, "check_shipping_status")["result"] == {
        "order_id": "A100",
        "carrier": "DHL",
        "tracking_number": "DHL123456",
        "status": "in_transit",
    }
    assert _tool_call(shipping, "get_rag_knowledge")["result"]
    assert shipping["state_snapshot"]["retrieved_knowledge"][0]["file_id"] == (
        "return_policy"
    )
    assert "DHL123456" in shipping["reply"]["body"]

    returns = by_case["return_policy"]
    rag_result = _tool_call(returns, "get_rag_knowledge")["result"]
    assert [item["file_id"] for item in rag_result] == ["return_policy"]
    assert "30 days" in returns["reply"]["body"]

    electronics = by_case["electronics_troubleshooting"]
    assert _slot_values(electronics) == {
        "product_model": "Airdog X5",
        "error_code": "E01",
        "symptom": "device will not start",
    }
    assert _tool_call(electronics, "get_rag_knowledge")["result"][0][
        "file_id"
    ] == "troubleshooting_3c"
    mcp_call = _tool_call(electronics, "lookup_product_manual")
    assert mcp_call["result"]["transport"] == "stdio"
    assert "filter compartment" in mcp_call["result"]["text"]
    assert "lookup_product_manual" in electronics["state_snapshot"][
        "mcp_tool_results"
    ]

    pet = by_case["pet_product_advice"]
    assert _slot_values(pet) == {"pet_type": "dog", "pet_weight": 12}
    assert "12 kg" in pet["reply"]["body"]
    assert "medium" in pet["reply"]["body"].lower()

    wig = by_case["wig_recommendation"]
    assert _slot_values(wig) == {
        "hair_type": "body wave",
        "color": "natural black",
        "length": "18 inches",
    }
    assert "18-inch" in wig["reply"]["body"]
    assert "natural black" in wig["reply"]["body"].lower()

    human = by_case["human_requested"]
    assert human["need_handoff_to_human"] is True
    assert human["handoff_type"] == "reply_handoff"
    assert human["reply"]["body"]

    high_risk = by_case["high_risk_complaint"]
    assert high_risk["need_handoff_to_human"] is True
    assert high_risk["handoff_type"] == "no_reply_handoff"
    assert high_risk["reply"]["body"] == ""

    german = by_case["german_reply"]["reply"]["body"]
    assert german.startswith("Guten Tag Frau Müller,")
    assert "Ihre Bestellung" in german
    assert german.endswith("Kundenservice")
    assert "Dear " not in german
    assert "Best regards" not in german

    dynamic_http = by_case["dynamic_http_tool"]
    http_call = _tool_call(dynamic_http, "lookup_after_sales_case")
    assert http_call["arguments"] == {"case_id": "RET-42"}
    assert http_call["result"] == {
        "case_id": "RET-42",
        "status": "approved",
        "source": "offline_fake_transport",
    }
    assert http_call["metadata"] == {
        "attempts": 1,
        "cache_hit": False,
        "status_code": 200,
    }
    assert "approved" in dynamic_http["reply"]["body"].lower()

    assert "Renée" in result.stdout
