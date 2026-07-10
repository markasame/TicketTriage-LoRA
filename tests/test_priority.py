from tickettriage.priority import (
    INTENT_BASE_PRIORITY,
    escalation_signals,
    label_priority,
    priority_target_text,
)
from tickettriage.prompts import INTENT_LABELS, PRIORITY_LABELS


def test_all_intents_have_base_priority():
    assert set(INTENT_BASE_PRIORITY) == set(INTENT_LABELS)
    assert set(INTENT_BASE_PRIORITY.values()) <= set(PRIORITY_LABELS)


def test_escalation_bumps_one_level():
    calm = "I would like to check my invoice from last month."
    angry = "I need my invoice immediately, this is urgent!"
    assert label_priority("check_invoice", calm)[0] == "low"
    assert label_priority("check_invoice", angry)[0] == "medium"


def test_urgent_is_capped():
    text = "unauthorized charge, fix this immediately, this is urgent!!!"
    priority, reason = label_priority("payment_issue", text)
    assert priority == "urgent"
    assert reason


def test_signals_detected():
    signals = escalation_signals("I was charged twice and I am furious")
    assert "charged twice" in signals
    assert "furious" in signals


def test_target_text_format():
    text = priority_target_text("track_order", "where is my package?")
    lines = text.splitlines()
    assert lines[0].startswith("priority: ")
    assert lines[1].startswith("reason: ")
    assert lines[0].split(": ")[1] in PRIORITY_LABELS


def test_deterministic():
    args = ("complaint", "I want to file a complaint about my order")
    assert label_priority(*args) == label_priority(*args)
