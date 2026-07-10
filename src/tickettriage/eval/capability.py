"""General-capability spot check (MMLU-lite).

A small bundled set of unambiguous multiple-choice questions across subjects,
run against base and fine-tuned models to confirm the fine-tune did not degrade
general reasoning. This is a smoke alarm, not a benchmark: 24 questions detect
gross regressions (chat-format collapse, catastrophic forgetting), nothing subtler.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_QUESTIONS_PATH = Path(__file__).parent / "mmlu_lite.json"

_PROMPT = (
    "Answer the following multiple-choice question. "
    "Reply with only the letter of the correct answer (A, B, C, or D).\n\n"
    "{question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}"
)


def load_questions() -> list[dict]:
    return json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))


def question_prompt(item: dict) -> str:
    a, b, c, d = item["choices"]
    return _PROMPT.format(question=item["q"], a=a, b=b, c=c, d=d)


def parse_answer(text: str) -> str | None:
    """Extract the first standalone A-D letter from model output."""
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None


def run_capability_check(backend) -> dict:
    """Run all questions through a Backend; returns accuracy and per-question detail."""
    questions = load_questions()
    detail = []
    correct = 0
    for item in questions:
        raw = backend.generate(question_prompt(item), max_new_tokens=10)
        picked = parse_answer(raw)
        ok = picked == item["answer"]
        correct += ok
        detail.append({"subject": item["subject"], "expected": item["answer"],
                       "picked": picked, "correct": ok})
    return {
        "accuracy": round(correct / len(questions), 4),
        "n": len(questions),
        "detail": detail,
    }
