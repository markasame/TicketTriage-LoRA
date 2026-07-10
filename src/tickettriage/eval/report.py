"""Render eval results (the JSON written by scripts/run_eval.py) as a Markdown report."""

from __future__ import annotations

from .classification import ClassificationReport, worst_classes


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(results: dict) -> str:
    """results: the dict produced by scripts/run_eval.py (see that file for the shape)."""
    base, ft = results["base"], results["finetuned"]
    lines = [
        "# TicketTriage-LoRA — Evaluation Report",
        "",
        f"Base model: `{results['base_model']}` · Fine-tuned: `{results['finetuned_model']}`",
        f"Test set: {results['n_test']} held-out tickets (never seen in training)",
        "",
        "## Headline comparison",
        "",
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        f"| Intent accuracy | {_pct(base['classification']['accuracy'])} | {_pct(ft['classification']['accuracy'])} |",
        f"| Intent macro-F1 | {base['classification']['macro_f1']:.3f} | {ft['classification']['macro_f1']:.3f} |",
        f"| Priority accuracy | {_pct(base['priority']['accuracy'])} | {_pct(ft['priority']['accuracy'])} |",
    ]
    if base.get("reply_metrics", {}).get("n"):
        brm, frm = base["reply_metrics"], ft["reply_metrics"]
        lines += [
            f"| Reply ROUGE-L vs reference | {brm['rouge_l']:.3f} | {frm['rouge_l']:.3f} |",
            f"| Reply token-F1 vs reference | {brm['token_f1']:.3f} | {frm['token_f1']:.3f} |",
            f"| Placeholder fidelity | {_pct(brm['placeholder_fidelity'])} | {_pct(frm['placeholder_fidelity'])} |",
        ]
    if base.get("judge", {}).get("n"):
        lines += [
            f"| Reply relevance (judge, 1-5) | {base['judge']['relevance']} | {ft['judge']['relevance']} |",
            f"| Reply tone (judge, 1-5) | {base['judge']['tone']} | {ft['judge']['tone']} |",
            f"| Reply correctness (judge, 1-5) | {base['judge']['correctness']} | {ft['judge']['correctness']} |",
        ]
    if base.get("capability"):
        lines.append(
            f"| MMLU-lite ({base['capability']['n']} q) | {_pct(base['capability']['accuracy'])} "
            f"| {_pct(ft['capability']['accuracy'])} |"
        )
    if base.get("cost", {}).get("n"):
        lines += [
            f"| Mean latency / ticket | {base['cost']['mean_latency_s']}s | {ft['cost']['mean_latency_s']}s |",
            f"| Cost / 1k tickets | ${base['cost']['usd_per_1k_tickets']} | ${ft['cost']['usd_per_1k_tickets']} |",
        ]
    lines += ["", "## Weakest intent classes (fine-tuned model)", ""]
    ft_report = ClassificationReport(
        accuracy=ft["classification"]["accuracy"],
        macro_f1=ft["classification"]["macro_f1"],
        per_class=ft["classification"]["per_class"],
        confusion={},
        n=ft["classification"]["n"],
    )
    for label, f1 in worst_classes(ft_report):
        stats = ft["classification"]["per_class"][label]
        lines.append(
            f"- `{label}`: F1 {f1:.2f} (precision {stats['precision']:.2f}, "
            f"recall {stats['recall']:.2f}, support {stats['support']})"
        )
    lines += [
        "",
        "## Confusion pairs (fine-tuned, top 5)",
        "",
    ]
    pairs = []
    for true_label, row in ft["classification"].get("confusion", {}).items():
        for pred_label, count in row.items():
            if pred_label != true_label:
                pairs.append((count, true_label, pred_label))
    for count, t, p in sorted(pairs, reverse=True)[:5]:
        lines.append(f"- `{t}` -> `{p}`: {count}x")
    lines += [
        "",
        "## Notes",
        "",
        "- Priority ground truth is rule-derived (see `src/tickettriage/priority.py`), "
        "not human-annotated; priority accuracy measures agreement with those rules.",
        "- Reply metrics are free reference-based scores (ROUGE-L / token-F1 against the "
        "dataset's reference reply, placeholder fidelity). Similarity to a single reference "
        "is a blunt instrument, but a fair relative signal between two models on the same "
        "tickets. The optional LLM judge (--judge) was not used unless judge rows appear above.",
        "- MMLU-lite is a 24-question smoke check for gross capability regressions, "
        "not a full benchmark.",
        "",
    ]
    return "\n".join(lines)
