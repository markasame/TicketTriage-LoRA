from tickettriage.eval.reply_metrics import placeholder_ok, rouge_l, score_replies, token_f1


def test_identical_texts_score_one():
    text = "We will look into your order {{Order Number}} right away."
    assert token_f1(text, text) == 1.0
    assert rouge_l(text, text) == 1.0


def test_disjoint_texts_score_zero():
    assert token_f1("apples bananas", "quantum flux") == 0.0
    assert rouge_l("apples bananas", "quantum flux") == 0.0


def test_rouge_l_is_order_sensitive_token_f1_is_not():
    ref = "please check your order status online"
    shuffled = "online status order your check please"
    assert token_f1(shuffled, ref) == 1.0
    assert rouge_l(shuffled, ref) < 1.0


def test_empty_candidate():
    assert token_f1("", "reference text") == 0.0
    assert rouge_l("", "reference text") == 0.0


def test_placeholders_normalize_to_same_token():
    a = "your order {{Order Number}} shipped"
    b = "your order {{Tracking Number}} shipped"
    assert token_f1(a, b) == 1.0


def test_placeholder_ok():
    ref = "We found your order {{Order Number}}."
    assert placeholder_ok("Sure, order {{Order Number}} is on its way.", ref)
    assert not placeholder_ok("Sure, order 58812 is on its way.", ref)
    # reference without placeholders never penalizes
    assert placeholder_ok("anything at all", "plain reference reply")


def test_score_replies_shape():
    rows = [
        {"reply": "a b c", "reference_reply": "a b c"},
        {"reply": "x y", "reference_reply": "a b {{Order Number}}"},
    ]
    summary = score_replies(rows)
    assert summary["n"] == 2
    assert 0 <= summary["rouge_l"] <= 1
    assert summary["placeholder_fidelity"] == 0.5
    assert score_replies([]) == {"n": 0}
