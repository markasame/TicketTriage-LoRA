# TicketTriage-LoRA — Evaluation Report

Base model: `hf:unsloth/Qwen3-8B-bnb-4bit` · Fine-tuned: `hf:unsloth/Qwen3-8B-bnb-4bit@models/tickettriage-lora`
Test set: 216 held-out tickets (never seen in training)

## Headline comparison

| Metric | Base | Fine-tuned |
|---|---|---|
| Intent accuracy | 80.1% | 96.3% |
| Intent macro-F1 | 0.782 | 0.963 |
| Priority accuracy | 37.5% | 63.9% |
| Reply ROUGE-L vs reference | 0.230 | 0.352 |
| Reply token-F1 vs reference | 0.395 | 0.502 |
| Placeholder fidelity | 60.2% | 85.2% |
| MMLU-lite (24 q) | 91.7% | 91.7% |
| Mean latency / ticket | 12.19s | 14.44s |
| Cost / 1k tickets | $1.49 | $1.765 |

## Weakest intent classes (fine-tuned model)

- `check_refund_policy`: F1 0.84 (precision 0.73, recall 1.00, support 8)
- `set_up_shipping_address`: F1 0.86 (precision 1.00, recall 0.75, support 8)
- `track_refund`: F1 0.86 (precision 1.00, recall 0.75, support 8)

## Confusion pairs (fine-tuned, top 5)

- `track_refund` -> `check_refund_policy`: 2x
- `set_up_shipping_address` -> `change_shipping_address`: 2x
- `get_refund` -> `check_refund_policy`: 1x
- `delivery_options` -> `place_order`: 1x
- `contact_human_agent` -> `contact_customer_service`: 1x

## Notes

- Priority ground truth is rule-derived (see `src/tickettriage/priority.py`), not human-annotated; priority accuracy measures agreement with those rules.
- Reply metrics are free reference-based scores (ROUGE-L / token-F1 against the dataset's reference reply, placeholder fidelity). Similarity to a single reference is a blunt instrument, but a fair relative signal between two models on the same tickets. The optional LLM judge (--judge) was not used unless judge rows appear above.
- MMLU-lite is a 24-question smoke check for gross capability regressions, not a full benchmark.
