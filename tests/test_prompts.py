from tickettriage.prompts import (
    INTENT_LABELS,
    chat_example,
    classify_prompt,
    parse_intent_output,
    parse_priority_output,
)


def test_classify_prompt_contains_all_labels():
    prompt = classify_prompt("where is my order?")
    for label in INTENT_LABELS:
        assert label in prompt
    assert "where is my order?" in prompt


def test_chat_example_shape():
    ex = chat_example("user text", "assistant text")
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][-1]["content"] == "assistant text"


def test_parse_intent_exact():
    assert parse_intent_output("track_order") == "track_order"
    assert parse_intent_output("  Track_Order \n") == "track_order"


def test_parse_intent_with_prose():
    assert parse_intent_output("The intent is track_order.") == "track_order"


def test_parse_intent_picks_first_mention():
    assert parse_intent_output("get_refund or maybe track_refund") == "get_refund"


def test_parse_intent_garbage_returns_none():
    assert parse_intent_output("I cannot classify this") is None


def test_parse_priority_wellformed():
    priority, reason = parse_priority_output("priority: high\nreason: Customer is blocked.")
    assert priority == "high"
    assert reason == "Customer is blocked."


def test_parse_priority_case_and_fallback():
    priority, _ = parse_priority_output("Priority: URGENT.\nreason: x")
    assert priority == "urgent"
    priority, _ = parse_priority_output("This looks like a medium priority issue to me")
    assert priority == "medium"


def test_parse_priority_garbage():
    priority, reason = parse_priority_output("no idea")
    assert priority is None
    assert reason == ""
