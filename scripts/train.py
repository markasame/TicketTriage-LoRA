"""QLoRA + DoRA fine-tuning for TicketTriage.

Stack: transformers Trainer + PEFT on a prequantized 4-bit NF4 base
(unsloth/Qwen3-8B-bnb-4bit — Qwen3 8B is the brief's ungated alternative to
Llama 3.1 8B). Completions-only loss is implemented by explicit prompt masking
(labels = -100 on prompt tokens) rather than TRL's assistant_only_loss, which
requires chat templates with {% generation %} markers that Llama/Qwen ship
without.

Config: LoRA r=16, alpha=16, DoRA on, all linear projections, lr 2e-4 cosine,
2 epochs, gradient checkpointing, eval every 25 steps, early stop on val loss.
Sized for an 8GB GPU (batch 1 x grad-accum 16, seq 1024); scales up unchanged
on a 4090.

Usage:
  python scripts/train.py                    # full run
  python scripts/train.py --max-steps 5      # smoke test
  python scripts/train.py --dry-run          # validate config/data, no torch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"  # prequantized NF4 mirror of Qwen/Qwen3-8B
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
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,  # default 8 spikes VRAM on 8GB cards
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
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
    dataloader_pin_memory=False,
)


def load_split(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    for i, row in enumerate(rows):
        msgs = row.get("messages")
        assert msgs and msgs[-1]["role"] == "assistant", f"bad example at {path}:{i + 1}"
    return rows


def dry_run() -> None:
    train_rows = load_split("data/train.jsonl")
    val_rows = load_split("data/val.jsonl")
    print(f"config OK - base={BASE_MODEL} lora={LORA_CONFIG}")
    print(f"data OK - train={len(train_rows)} val={len(val_rows)}")


def encode_rows(rows: list[dict], tokenizer) -> list[dict]:
    """Tokenize chat examples with prompt tokens masked out of the loss.

    Prompt and completion are tokenized separately and concatenated, so the
    mask boundary is exact (re-tokenizing the concatenated string can shift
    token boundaries). enable_thinking=False keeps Qwen3's template in
    non-thinking mode; the kwarg is ignored by templates that don't use it.
    """
    encoded = []
    skipped = 0
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            row["messages"][:-1],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        completion_text = row["messages"][-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:MAX_SEQ_LEN]
        if len(prompt_ids) >= MAX_SEQ_LEN:  # no completion tokens left to learn from
            skipped += 1
            continue
        labels = [-100] * len(prompt_ids) + completion_ids
        encoded.append({"input_ids": input_ids, "labels": labels[: len(input_ids)]})
    if skipped:
        print(f"skipped {skipped} examples with prompt >= {MAX_SEQ_LEN} tokens")
    return encoded


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


def train(max_steps: int | None, base_model: str, no_dora: bool, resume: bool = False) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_data = encode_rows(load_split("data/train.jsonl"), tokenizer)
    val_data = encode_rows(load_split("data/val.jsonl"), tokenizer)
    print(f"encoded: train={len(train_data)} val={len(val_data)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # device_map={"": 0}: everything on GPU 0. "auto" silently dispatches layers to
    # CPU when the desktop is using VRAM, which bnb-4bit rejects; forcing GPU lets
    # Windows WDDM page into shared memory instead (slower, but it runs).
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = dict(LORA_CONFIG)
    if no_dora:
        lora_cfg["use_dora"] = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora_cfg))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAIN_CONFIG,
        **({"max_steps": max_steps} if max_steps else {}),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=PadCollator(tokenizer.pad_token_id or tokenizer.eos_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    ckpt = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint

        ckpt = get_last_checkpoint(OUTPUT_DIR)
        if ckpt:
            print(f"resuming from {ckpt}")
    result = trainer.train(resume_from_checkpoint=ckpt)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log_path = Path(OUTPUT_DIR) / "training_log.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"done - train loss {result.training_loss:.4f}; log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--no-dora", action="store_true", help="plain LoRA (lower VRAM)")
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        train(args.max_steps, args.base_model, args.no_dora, args.resume)


if __name__ == "__main__":
    main()
