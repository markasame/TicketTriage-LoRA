import json
import random

from scripts.prepare_data import TEST_PER_INTENT, build_splits, curate, synthetic_rows
from tickettriage.prompts import INTENT_LABELS


def _splits():
    by_intent = curate(synthetic_rows())
    return build_splits(by_intent, random.Random(0))


def test_curate_filters_and_dedupes():
    rows = synthetic_rows()
    rows.append(rows[0].copy())  # duplicate instruction
    rows.append({"instruction": "short", "intent": "review", "response": "x" * 100})
    rows.append({"instruction": "y" * 50, "intent": "not_a_label", "response": "x" * 100})
    by_intent = curate(rows)
    assert set(by_intent) <= set(INTENT_LABELS)
    total = sum(len(v) for v in by_intent.values())
    assert total == sum(len(v) for v in curate(synthetic_rows()).values())  # extras rejected


def test_split_sizes_and_shape():
    train, val, test = _splits()
    assert len(test) == TEST_PER_INTENT * len(INTENT_LABELS)
    assert len(val) >= len(train) // 10 - 1
    assert 800 <= len(train) + len(val) <= 1500  # brief: curated size, quality over volume

    ex = train[0]
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["task"] in {"classify", "priority", "reply"}

    t = test[0]
    assert set(t) == {"ticket", "intent", "priority", "priority_reason", "reference_reply"}


def test_no_test_leakage_into_train():
    train, val, test = _splits()
    test_tickets = {t["ticket"] for t in test}
    for ex in train + val:
        user_msg = ex["messages"][1]["content"]
        for ticket in test_tickets:
            assert ticket not in user_msg


def test_all_three_tasks_present():
    train, val, _ = _splits()
    tasks = {ex["task"] for ex in train + val}
    assert tasks == {"classify", "priority", "reply"}


def test_examples_are_json_serializable():
    train, _, _ = _splits()
    json.dumps(train[:5])
