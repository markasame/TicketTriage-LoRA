import pytest

from tickettriage.eval.classification import evaluate_classification, worst_classes


def test_perfect_predictions():
    y = ["a", "b", "a", "c"]
    report = evaluate_classification(y, y)
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.n_unparseable == 0


def test_known_values():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    report = evaluate_classification(y_true, y_pred)
    assert report.accuracy == 0.75
    # class a: precision 1/1, recall 1/2, f1 = 2/3
    assert report.per_class["a"]["f1"] == pytest.approx(0.6667, abs=1e-3)
    # class b: precision 2/3, recall 2/2, f1 = 0.8
    assert report.per_class["b"]["f1"] == pytest.approx(0.8, abs=1e-3)
    assert report.confusion["a"]["b"] == 1


def test_unparseable_counts_as_wrong():
    report = evaluate_classification(["a", "b"], ["a", None])
    assert report.accuracy == 0.5
    assert report.n_unparseable == 1
    assert report.confusion["b"]["<unparseable>"] == 1


def test_predicted_only_label_included():
    report = evaluate_classification(["a", "a"], ["a", "z"])
    assert "z" in report.per_class
    assert report.per_class["z"]["support"] == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_classification(["a"], [])


def test_empty_raises():
    with pytest.raises(ValueError):
        evaluate_classification([], [])


def test_worst_classes():
    y_true = ["a"] * 5 + ["b"] * 5
    y_pred = ["a"] * 5 + ["a"] * 5  # b always wrong
    report = evaluate_classification(y_true, y_pred)
    worst = worst_classes(report, k=1)
    assert worst[0][0] == "b"
    assert worst[0][1] == 0.0


def test_to_dict_roundtrip():
    report = evaluate_classification(["a", "b"], ["a", "b"])
    d = report.to_dict()
    assert d["accuracy"] == 1.0
    assert "confusion" in d and d["per_class"]["a"]["support"] == 1
