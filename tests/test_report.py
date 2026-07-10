from tickettriage.eval.report import render_report


def _side(acc: float) -> dict:
    return {
        "classification": {
            "accuracy": acc,
            "macro_f1": acc,
            "n": 10,
            "per_class": {
                "track_order": {"precision": acc, "recall": acc, "f1": acc, "support": 5},
                "complaint": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 5},
            },
            "confusion": {"complaint": {"track_order": 3, "complaint": 2}},
        },
        "priority": {"accuracy": acc, "macro_f1": acc},
        "judge": {"n": 10, "relevance": 4.0, "tone": 4.2, "correctness": 3.8, "mean": 4.0},
        "capability": {"accuracy": 0.8, "n": 24},
        "cost": {
            "n": 10, "mean_latency_s": 2.5, "median_latency_s": 2.4, "p95_latency_s": 3.0,
            "gpu_usd_per_hour": 0.44, "usd_per_ticket": 0.0003, "usd_per_1k_tickets": 0.306,
        },
    }


def test_render_report_contains_key_sections():
    results = {
        "base_model": "base", "finetuned_model": "ft", "n_test": 10,
        "base": _side(0.6), "finetuned": _side(0.9),
    }
    report = render_report(results)
    assert "Headline comparison" in report
    assert "90.0%" in report and "60.0%" in report
    assert "complaint" in report          # weakest class surfaced
    assert "`complaint` -> `track_order`: 3x" in report
    assert "rule-derived" in report       # honesty note about priority labels
