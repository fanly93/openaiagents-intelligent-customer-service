import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
DEMO_SCRIPT = PROJECT_ROOT / "scripts/run_demo_cases.py"


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
    responses = [json.loads(line) for line in lines]

    assert len(responses) == 2
    by_request_id = {
        response["request_id"]: response
        for response in responses
    }

    normal = by_request_id["demo_order"]
    assert normal["status"] == "success"
    assert normal["need_handoff_to_human"] is False
    assert normal["handoff_type"] == "no_handoff"
    assert normal["error"] is None

    handoff = by_request_id["demo_handoff"]
    assert handoff["status"] == "success"
    assert handoff["need_handoff_to_human"] is True
    assert handoff["handoff_type"] == "reply_handoff"
    assert handoff["error"] is None

    assert "Renée" in result.stdout
