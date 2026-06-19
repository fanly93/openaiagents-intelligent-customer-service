import pytest
from fastapi.testclient import TestClient

from scripts.run_mock_http_server import app


client = TestClient(app)


def test_mock_order_returns_stable_response_contract():
    response = client.post("/mock/order", json={"order_id": "ORDER-100"})

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "data": {
            "order_id": "ORDER-100",
            "status": "shipped",
        },
    }


def test_mock_logistics_returns_stable_response_contract():
    response = client.post("/mock/logistics", json={"order_id": "ORDER-100"})

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "data": {
            "order_id": "ORDER-100",
            "carrier": "DHL",
            "tracking_number": "DHL123456",
            "status": "in_transit",
        },
    }


def test_mock_refund_policy_returns_stable_response_contract():
    response = client.get("/mock/refund-policy")

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "data": {
            "policy": "Returns are accepted within 30 days.",
        },
    }


@pytest.mark.parametrize("path", ["/mock/order", "/mock/logistics"])
@pytest.mark.parametrize("payload", [{}, {"order_id": ""}, {"order_id": "   "}])
def test_order_endpoints_reject_missing_or_blank_order_id(path, payload):
    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "order_id"]
