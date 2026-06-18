from app.harness.reply_post_processor import ReplyPostProcessor


def test_reply_post_processor_removes_react_markers():
    processor = ReplyPostProcessor()
    cleaned = processor.clean(
        "THINK: I need a tool\nACTION: call tool\nDear customer,\nWe can help."
    )

    assert "THINK:" not in cleaned
    assert "ACTION:" not in cleaned
    assert "Dear customer" in cleaned
