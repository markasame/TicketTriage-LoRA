"""Gradio demo: paste a ticket, see base vs fine-tuned triage side by side.

Deployed to Hugging Face Spaces (CPU free tier). Two modes:

- LIVE mode: if a triage backend is reachable (env BASE_BACKEND / FT_BACKEND,
  e.g. "ollama:llama3.1:8b" and "ollama:tickettriage"), tickets are triaged live.
- SHOWCASE mode (default on Spaces free tier, where an 8B model is impractical):
  serves precomputed base-vs-fine-tuned outputs for real held-out test tickets
  from results/eval_results.json, so the comparison shown is the *actual* eval.

Eval scores are rendered on the page either way, from results/eval_results.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_PATH = ROOT / "results" / "eval_results.json"


def load_results() -> dict | None:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return None


def scores_markdown(results: dict | None) -> str:
    if not results:
        return "*No eval results found yet — run `scripts/run_eval.py` to populate this table.*"
    base, ft = results["base"], results["finetuned"]
    rows = [
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        f"| Intent accuracy | {base['classification']['accuracy']:.1%} "
        f"| **{ft['classification']['accuracy']:.1%}** |",
        f"| Intent macro-F1 | {base['classification']['macro_f1']:.3f} "
        f"| **{ft['classification']['macro_f1']:.3f}** |",
        f"| Priority accuracy | {base['priority']['accuracy']:.1%} "
        f"| **{ft['priority']['accuracy']:.1%}** |",
    ]
    if base.get("judge", {}).get("n"):
        rows.append(
            f"| Reply quality (judge mean, 1-5) | {base['judge']['mean']} "
            f"| **{ft['judge']['mean']}** |"
        )
    if base.get("capability"):
        rows.append(
            f"| MMLU-lite | {base['capability']['accuracy']:.1%} "
            f"| {ft['capability']['accuracy']:.1%} |"
        )
    rows.append(f"\n*Evaluated on {results['n_test']} held-out tickets. "
                f"Judge: Claude with a fixed rubric, blind to model identity.*")
    return "\n".join(rows)


def make_live_backends():
    base_spec = os.environ.get("BASE_BACKEND")
    ft_spec = os.environ.get("FT_BACKEND")
    if not (base_spec and ft_spec):
        return None, None
    from tickettriage.inference import make_backend

    try:
        return make_backend(base_spec), make_backend(ft_spec)
    except Exception:
        return None, None


def format_triage(intent, priority, reason, reply) -> str:
    return (
        f"**Intent:** `{intent}`\n\n**Priority:** `{priority}` — {reason}\n\n"
        f"**Draft reply:**\n\n{reply}"
    )


def build_app() -> gr.Blocks:
    results = load_results()
    base_backend, ft_backend = make_live_backends()
    live = base_backend is not None

    showcase: list[dict] = []
    if results and not live:
        base_replies = {r["ticket"]: r["reply"] for r in results["base"].get("replies", [])}
        for row in results["finetuned"].get("replies", []):
            if row["ticket"] in base_replies:
                showcase.append(
                    {"ticket": row["ticket"], "base": base_replies[row["ticket"]],
                     "ft": row["reply"]}
                )

    def run(ticket: str):
        ticket = (ticket or "").strip()
        if not ticket:
            return "Enter a ticket first.", ""
        if live:
            b, f = base_backend.triage(ticket), ft_backend.triage(ticket)
            return (
                format_triage(b.intent, b.priority, b.priority_reason, b.draft_reply),
                format_triage(f.intent, f.priority, f.priority_reason, f.draft_reply),
            )
        # showcase mode: nearest precomputed ticket by naive token overlap
        if not showcase:
            return ("Showcase data missing — deploy results/eval_results.json.",) * 2
        words = set(ticket.lower().split())
        best = max(showcase, key=lambda r: len(words & set(r["ticket"].lower().split())))
        note = f"*(showcase mode — closest held-out test ticket: “{best['ticket']}”)*\n\n"
        return note + best["base"], note + best["ft"]

    examples = [row["ticket"] for row in showcase[:6]] or [
        "I was charged twice for order 5531 and nobody answers the phone. Fix this now!",
        "hi, how do I change the shipping address on my last order?",
    ]

    with gr.Blocks(title="TicketTriage-LoRA") as demo:
        gr.Markdown("# 🎫 TicketTriage-LoRA\n"
                    "QLoRA+DoRA fine-tune of Llama 3.1 8B for support-ticket triage — "
                    "side-by-side with the untuned base model.")
        gr.Markdown(scores_markdown(results))
        mode = "live inference" if live else "showcase (precomputed eval outputs)"
        gr.Markdown(f"**Mode:** {mode}")
        ticket_box = gr.Textbox(label="Support ticket", lines=4,
                                placeholder="Paste a customer message...")
        btn = gr.Button("Triage", variant="primary")
        with gr.Row():
            base_out = gr.Markdown(label="Base model")
            ft_out = gr.Markdown(label="Fine-tuned model")
        gr.Examples(examples=examples, inputs=ticket_box)
        btn.click(run, inputs=ticket_box, outputs=[base_out, ft_out])
        ticket_box.submit(run, inputs=ticket_box, outputs=[base_out, ft_out])
    return demo


if __name__ == "__main__":
    build_app().launch()
