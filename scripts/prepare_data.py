"""Download and curate the Bitext customer-support dataset into training JSONL.

Curation philosophy: quality over volume. From ~27k raw rows we keep ~1.2k
balanced, deduplicated, length-filtered examples across three task types
(intent classification, priority scoring, reply drafting), plus a held-out
test set of tickets that never appear in training.

Outputs (JSONL):
  data/train.jsonl  - chat-format training examples (90%)
  data/val.jsonl    - chat-format validation examples (10%)
  data/test.jsonl   - held-out raw tickets with gold intent + silver priority

Usage:
  python scripts/prepare_data.py               # real dataset from HF hub
  python scripts/prepare_data.py --synthetic   # tiny fake dataset (CI smoke test)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tickettriage.priority import label_priority, priority_target_text  # noqa: E402
from tickettriage.prompts import (  # noqa: E402
    INTENT_LABELS,
    chat_example,
    classify_prompt,
    priority_prompt,
    reply_prompt,
)

HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SEED = 42

TEST_PER_INTENT = 8       # held-out tickets per intent (27 intents -> ~216 test tickets)
TRAIN_PER_INTENT = 45     # curated training tickets per intent (~1215 examples)
TASK_CYCLE = ["classify", "reply", "priority", "classify", "reply"]  # 40/40/20 mix


def load_raw_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(HF_DATASET, repo_type="dataset")
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"No CSV found in {HF_DATASET}; files: {files}")
    path = hf_hub_download(HF_DATASET, csv_files[0], repo_type="dataset")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def synthetic_rows(n_per_intent: int = 60) -> list[dict]:
    """Fake rows for offline smoke tests: unique instructions, valid schema."""
    rows = []
    for intent in INTENT_LABELS:
        for i in range(n_per_intent):
            rows.append(
                {
                    "flags": "B",
                    "instruction": f"synthetic ticket {i} about {intent.replace('_', ' ')}: "
                    f"I need help with this issue, please assist me promptly (case {i}).",
                    "category": "SYNTHETIC",
                    "intent": intent,
                    "response": f"Certainly! Here is help with {intent.replace('_', ' ')} "
                    f"for your order {{{{Order Number}}}}. " + "We will resolve this. " * 5,
                }
            )
    return rows


def curate(rows: list[dict]) -> dict[str, list[dict]]:
    """Filter, dedupe, and bucket rows by intent."""
    seen: set[str] = set()
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        intent = (row.get("intent") or "").strip()
        if intent not in INTENT_LABELS:
            continue
        if not (20 <= len(instruction) <= 500 and 50 <= len(response) <= 2000):
            continue
        key = instruction.lower()
        if key in seen:
            continue
        seen.add(key)
        by_intent[intent].append({"ticket": instruction, "intent": intent, "response": response})
    return by_intent


def build_splits(by_intent: dict[str, list[dict]], rng: random.Random) -> tuple[list, list, list]:
    train_examples: list[dict] = []
    test_rows: list[dict] = []
    for intent in sorted(by_intent):
        bucket = by_intent[intent][:]
        rng.shuffle(bucket)
        test_bucket = bucket[:TEST_PER_INTENT]
        train_bucket = bucket[TEST_PER_INTENT : TEST_PER_INTENT + TRAIN_PER_INTENT]

        for row in test_bucket:
            priority, reason = label_priority(intent, row["ticket"])
            test_rows.append(
                {
                    "ticket": row["ticket"],
                    "intent": intent,
                    "priority": priority,
                    "priority_reason": reason,
                    "reference_reply": row["response"],
                }
            )

        for i, row in enumerate(train_bucket):
            task = TASK_CYCLE[i % len(TASK_CYCLE)]
            if task == "classify":
                example = chat_example(classify_prompt(row["ticket"]), intent)
            elif task == "priority":
                example = chat_example(
                    priority_prompt(row["ticket"]), priority_target_text(intent, row["ticket"])
                )
            else:
                example = chat_example(reply_prompt(row["ticket"]), row["response"])
            example["task"] = task
            example["intent"] = intent
            train_examples.append(example)

    rng.shuffle(train_examples)
    n_val = max(1, len(train_examples) // 10)
    return train_examples[n_val:], train_examples[:n_val], test_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="offline smoke-test data")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows = synthetic_rows() if args.synthetic else load_raw_rows()
    print(f"raw rows: {len(rows)}")

    by_intent = curate(rows)
    kept = sum(len(v) for v in by_intent.values())
    print(f"after curation: {kept} rows across {len(by_intent)} intents")

    train, val, test = build_splits(by_intent, rng)
    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    task_counts = defaultdict(int)
    for ex in train + val:
        task_counts[ex["task"]] += 1
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f"task mix (train+val): {dict(task_counts)}")


if __name__ == "__main__":
    main()
