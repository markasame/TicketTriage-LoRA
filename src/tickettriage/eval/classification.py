"""Classification metrics: accuracy, per-class precision/recall/F1, confusion matrix.

Implemented in pure Python so the eval harness (and CI) has no sklearn dependency.
Unparseable predictions are passed as None and count as wrong for every class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]  # label -> {precision, recall, f1, support}
    confusion: dict[str, Counter] = field(repr=False)  # true label -> Counter of predicted
    n: int = 0
    n_unparseable: int = 0

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n": self.n,
            "n_unparseable": self.n_unparseable,
            "per_class": self.per_class,
            "confusion": {t: dict(c) for t, c in self.confusion.items()},
        }


def evaluate_classification(
    y_true: list[str], y_pred: list[str | None]
) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("empty evaluation set")

    labels = sorted(set(y_true) | {p for p in y_pred if p is not None})
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    unparseable = 0
    for t, p in zip(y_true, y_pred):
        key = p if p is not None else "<unparseable>"
        confusion[t][key] += 1
        if p == t:
            correct += 1
        if p is None:
            unparseable += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_sum = 0.0
    true_counts = Counter(y_true)
    pred_counts = Counter(p for p in y_pred if p is not None)
    for label in labels:
        tp = confusion[label][label]
        support = true_counts[label]
        precision = tp / pred_counts[label] if pred_counts[label] else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_sum += f1

    return ClassificationReport(
        accuracy=round(correct / len(y_true), 4),
        macro_f1=round(f1_sum / len(labels), 4),
        per_class=per_class,
        confusion=confusion,
        n=len(y_true),
        n_unparseable=unparseable,
    )


def worst_classes(report: ClassificationReport, k: int = 3) -> list[tuple[str, float]]:
    """The k lowest-F1 classes with nonzero support — used for the honest-failure section."""
    scored = [
        (label, stats["f1"])
        for label, stats in report.per_class.items()
        if stats["support"] > 0
    ]
    return sorted(scored, key=lambda x: x[1])[:k]
