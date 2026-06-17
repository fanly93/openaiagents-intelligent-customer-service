from app.api.compatibility import convert_emailv4_payload


def test_emailv4_payload_maps_to_customer_service_request():
    req = convert_emailv4_payload(
        {
            "corp": "3686",
            "content": "Where is my order #100?",
            "customer_email": "buyer@example.com",
            "customer_name": "Ada",
            "uuid": "user_1",
            "webhook": "https://example.com/hook",
            "contexts": [{"role": "user", "content": "Previous message"}],
        }
    )

    assert req.tenant_id == "3686"
    assert req.customer.email == "buyer@example.com"
    assert req.customer.name == "Ada"
    assert req.customer.id == "user_1"
    assert req.webhook_url == "https://example.com/hook"
    assert req.contexts[0].content == "Previous message"
