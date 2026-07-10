"""QLoRA + DoRA fine-tuning for TicketTriage.

Primary path: Unsloth (fastest single-GPU QLoRA). Fallback: HF TRL + PEFT if
Unsloth is unavailable or rejects the base model. Both paths share the config:

  4-bit NF4 base - LoRA r=16, alpha=16, DoRA on, target all linear projections
  lr 2e-4 cosine - 2 epochs - gradient checkpointing - completions-only loss
  eval on val split every 25 steps - early stop when val loss climbs

Run on a >=12GB GPU (RunPod RTX 4090: scripts/runpod_run.sh drives the whole
pipeline). CI only runs --dry-run (config + data validation, no torch).

Usage:
  python scripts/train.py                          # full run
  python scripts/train.py --max-steps 5            # smoke test
  python scripts/train.py --dry-run                # validate config/data only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "models/tickettriage-lora"

LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    use_dora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

TRAIN_CONFIG = dict(
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    load_best_model_at_end=True,           # early-stop companion: keep best val-loss ckpt
    metric_for_best_model="eval_loss",
    bf16=True,
    optim="adamw_8bit",
    report_to="none",
    seed=42,
)


def load_split(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    for i, row in enumerate(rows):
        msgs = row.get("messages")
        assert msgs and msgs[-1]["role"] == "assistant", f"bad example at {path}:{i + 1}"
    return rows


def dry_run() -> None:
    train = load_split("data/train.jsonl")
    val = load_split("data/val.jsonl")
    print(f"config OK - base={BASE_MODEL} lora={LORA_CONFIG}")
    print(f"data OK - train={len(train)} val={len(val)}")


def build_dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def train(max_steps: int | None) -> None:
    train_rows, val_rows = load_split("data/train.jsonl"), load_split("data/val.jsonl")
    train_ds, val_ds = build_dataset(train_rows), build_dataset(val_rows)

    use_unsloth = True
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        use_unsloth = False
        print("Unsloth not available - falling back to TRL + PEFT")

    if use_unsloth:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
            )
            model = FastLanguageModel.get_peft_model(
                model,
                use_gradient_checkpointing="unsloth",
                random_state=42,
                **LORA_CONFIG,
            )
        except Exception as exc:
            print(f"Unsloth failed on {BASE_MODEL} ({exc}) - falling back to TRL + PEFT")
            use_unsloth = False

    if not use_unsloth:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        model.print_trainable_parameters()

    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LEN,
        assistant_only_loss=True,  # train on completions only (mask the prompt)
        **TRAIN_CONFIG,
        **({"max_steps": max_steps} if max_steps else {}),
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log_path = Path(OUTPUT_DIR) / "training_log.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"done - train loss {result.training_loss:.4f}; log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        train(args.max_steps)


if __name__ == "__main__":
    main()
