"""Before/after evaluation: base model vs fine-tuned adapter on the held-out test set.

Produces results/eval_results.json and results/eval_report.md covering:
  - intent classification: accuracy, per-class F1, confusion matrix
  - priority scoring: accuracy vs rule-derived labels
  - reply quality: blind Claude-judge rubric scores (needs ANTHROPIC_API_KEY)
  - latency + $ per ticket
  - MMLU-lite general-capability spot check

Usage (on the GPU box after training):
  python scripts/run_eval.py \
      --base hf:meta-llama/Llama-3.1-8B-Instruct \
      --finetuned hf:meta-llama/Llama-3.1-8B-Instruct@models/tickettriage-lora

Any backend spec from tickettriage.inference works (echo / ollama:… / hf:…).
Use --limit and --skip-judge for quick partial runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tickettriage.eval.capability import run_capability_check  # noqa: E402
from tickettriage.eval.classification import evaluate_classification  # noqa: E402
from tickettriage.eval.costs import DEFAULT_GPU_USD_PER_HOUR, cost_summary  # noqa: E402
from tickettriage.eval.judge import judge_batch, summarize_scores  # noqa: E402
from tickettriage.eval.report import render_report  # noqa: E402
from tickettriage.inference import make_backend  # noqa: E402


def eval_model(spec: str, test_rows: list[dict], gpu_rate: float, skip_capability: bool) -> dict:
    backend = make_backend(spec)
    print(f"\n=== {backend.name}: triaging {len(test_rows)} tickets ===")
    predictions = []
    for i, row in enumerate(test_rows):
        result = backend.triage(row["ticket"])
        predictions.append(result)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(test_rows)}")

    classification = evaluate_classification(
        [r["intent"] for r in test_rows], [p.intent for p in predictions]
    )
    priority = evaluate_classification(
        [r["priority"] for r in test_rows], [p.priority for p in predictions]
    )
    out = {
        "name": backend.name,
        "classification": classification.to_dict(),
        "priority": {"accuracy": priority.accuracy, "macro_f1": priority.macro_f1},
        "cost": cost_summary([p.latency_s for p in predictions], gpu_rate),
        "replies": [
            {"ticket": r["ticket"], "reply": p.draft_reply}
            for r, p in zip(test_rows, predictions)
        ],
    }
    if not skip_capability:
        print("  running MMLU-lite capability check...")
        out["capability"] = run_capability_check(backend)
    return out


def run_judge(results: dict) -> None:
    import anthropic

    client = anthropic.Anthropic()
    for side in ("base", "finetuned"):
        rows = results[side]["replies"]
        print(f"judging {len(rows)} {side} replies...")
        scores = judge_batch(client, rows, "reply")
        results[side]["judge"] = summarize_scores(scores)
        results[side]["judge_detail"] = [s.model_dump() if s else None for s in scores]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="backend spec for the base model")
    parser.add_argument("--finetuned", required=True, help="backend spec for the tuned model")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tickets")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-capability", action="store_true")
    parser.add_argument("--gpu-usd-per-hour", type=float, default=DEFAULT_GPU_USD_PER_HOUR)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    test_rows = [
        json.loads(line)
        for line in Path(args.test_file).read_text(encoding="utf-8").splitlines()
    ]
    if args.limit:
        test_rows = test_rows[: args.limit]

    results = {
        "base_model": args.base,
        "finetuned_model": args.finetuned,
        "n_test": len(test_rows),
        "base": eval_model(args.base, test_rows, args.gpu_usd_per_hour, args.skip_capability),
        "finetuned": eval_model(
            args.finetuned, test_rows, args.gpu_usd_per_hour, args.skip_capability
        ),
    }
    if not args.skip_judge:
        run_judge(results)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = render_report(results)
    (out / "eval_report.md").write_text(report, encoding="utf-8")
    print(f"\nwrote {out / 'eval_results.json'} and {out / 'eval_report.md'}\n")
    print(report)


if __name__ == "__main__":
    main()
