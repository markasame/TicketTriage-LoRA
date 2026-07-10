# 🎫 TicketTriage-LoRA

A QLoRA + DoRA fine-tune of **Qwen3 8B** for customer-support ticket triage —
intent classification, priority scoring, and reply drafting from a single adapter — with a
rigorous **before/after evaluation** against the untuned base model.

**[▶ Live demo on HF Spaces](https://huggingface.co/spaces/CHANGEME/tickettriage-lora)** · *(demo GIF goes here after the first training run)*

## Results

> ⚠️ **Pending first training run.** The whole project is **100% free to reproduce** — no
> paid APIs, no gated models. Training + eval run unattended on any ≥8GB consumer GPU
> (`python scripts/train.py` then `scripts/run_eval.py`; on a busy Windows desktop,
> `scripts/local_supervisor.ps1` waits for free VRAM and drives both). The run writes
> `results/eval_report.md`; paste the headline table here:

| Metric (216 held-out tickets) | Base Qwen3 8B | Fine-tuned |
|---|---|---|
| Intent accuracy (27 classes) | *tbd* | *tbd* |
| Intent macro-F1 | *tbd* | *tbd* |
| Priority accuracy | *tbd* | *tbd* |
| Reply ROUGE-L vs reference | *tbd* | *tbd* |
| Reply token-F1 vs reference | *tbd* | *tbd* |
| Placeholder fidelity | *tbd* | *tbd* |
| MMLU-lite (capability regression check) | *tbd* | *tbd* |
| Mean latency / ticket | *tbd* | *tbd* |
| $ / 1k tickets (RTX 4090 @ $0.44/hr) | *tbd* | *tbd* |

**Base model choice:** the brief allowed Llama 3.1 8B Instruct or Qwen3 8B; this run uses
**Qwen3 8B** (via the prequantized `unsloth/Qwen3-8B-bnb-4bit`) because it is ungated — the
whole pipeline reproduces without a Hugging Face access request. Swap `--base-model` to use
Llama 3.1 instead.

### Why fine-tune instead of just prompting the base model?

The base model already "knows" customer support — what it can't do reliably from a prompt is
(1) emit **exactly one of 27 intent labels** with no prose around it, (2) apply a **consistent
priority policy** instead of inventing its own severity scale per ticket, and (3) keep
`{{Order Number}}`-style placeholders intact instead of hallucinating concrete order details.
The fine-tune bakes the output contract and the policy into the weights, which is what moves
classification accuracy and judge-scored correctness — the instructions in the prompt are
identical for both models in the eval.

### An honest failure case

*Fill in from `results/eval_report.md` → "Weakest intent classes" + "Confusion pairs" after
the run.* Expected from the label space: `track_refund` vs `get_refund` — tickets like
*"I still haven't gotten my money back"* sit exactly on the boundary between *asking where a
refund is* and *requesting one*, and the model picks the wrong side when the ticket doesn't
say whether a refund was already initiated.

## What's in the box

```
scripts/prepare_data.py   Bitext 27k → 1,215 curated examples (dedup, length-filter,
                          class-balanced) + 216 held-out test tickets. Committed in data/.
scripts/train.py          QLoRA+DoRA via transformers+PEFT: 4-bit NF4, r=16, α=16,
                          all-linear targets, lr 2e-4, 2 epochs, completions-only loss
                          (explicit prompt masking), grad checkpointing, early stop on
                          val loss. Sized to fit an 8GB GPU; faster on a 4090.
scripts/run_eval.py       The deliverable that matters: accuracy/F1/confusion for intent,
                          priority accuracy, free reference-based reply metrics (ROUGE-L,
                          token-F1, placeholder fidelity; optional --judge LLM rubric),
                          latency + $/ticket, MMLU-lite regression check.
scripts/export_gguf.py    Merge adapter → GGUF q4_k_m → Ollama Modelfile.
scripts/runpod_run.sh     All of the above as one unattended pipeline.
src/tickettriage/         Prompts, priority policy, inference backends, FastAPI service.
app/app.py                Gradio demo (HF Spaces): base vs fine-tuned side by side,
                          eval scores on the page.
```

### Three tasks, one adapter

Training data is ChatML with a shared system prompt; each curated ticket becomes one of:

| Task | Target | Share |
|---|---|---|
| Intent classification | bare label from the 27-label set | 40% |
| Priority scoring | `priority: <level>` + one-sentence reason | 20% |
| Reply drafting | full reply, placeholders preserved | 40% |

**A note on DoRA:** the config supports `use_dora=True` and it trains correctly, but on
bnb-4bit quantized weights DoRA dequantizes the full base weight every forward to compute
weight norms — measured **~14 min/optimizer-step vs ~1–2 min for plain LoRA** on an RTX
3060 Ti. The shipped local run therefore uses plain LoRA (`--no-dora`); "free quality
upgrade" only holds on unquantized or Unsloth-fused setups.

**Priority labels are rule-derived** (`src/tickettriage/priority.py`: base priority per intent,
bumped one level on urgency language). They're deterministic, documented "silver" labels — the
eval measures agreement with the policy, and this README won't pretend they're human annotations.

## Reproduce

```bash
# 1. Data (already committed; regenerate with)
python scripts/prepare_data.py

# 2. Train — free, fits an 8GB GPU (batch 1 × grad-accum 16)
python scripts/train.py --no-dora
python scripts/run_eval.py --base "hf:unsloth/Qwen3-8B-bnb-4bit" \
    --finetuned "hf:unsloth/Qwen3-8B-bnb-4bit@models/tickettriage-lora"
# busy Windows desktop? scripts/local_supervisor.ps1 waits for VRAM and runs both
# rented GPU instead: bash scripts/runpod_run.sh (optional, ~$2 on an RTX 4090)

# 3. Serve locally
ollama create tickettriage -f models/gguf/Modelfile
TRIAGE_BACKEND="ollama:tickettriage" uvicorn tickettriage.api:app
curl -X POST localhost:8000/triage -H 'Content-Type: application/json' \
     -d '{"ticket": "I was charged twice and I am furious"}'
# → {"intent": "payment_issue", "priority": "urgent", "priority_reason": ..., "draft_reply": ...}

# 4. Demo → HF Spaces
python scripts/deploy_space.py --space <you>/tickettriage-lora
```

### Eval design notes

- **Held-out test set**: 216 tickets (8 per intent) split off *before* training-example
  construction; a test assertion guards against leakage.
- **Reply quality (free, default)**: ROUGE-L and token-F1 against the dataset's reference
  reply, plus placeholder fidelity (does the reply keep `{{Order Number}}`-style slots
  instead of inventing concrete details?). Similarity to a single reference is a blunt
  instrument — a good reply worded differently scores lower than a mediocre parrot — but
  it is deterministic, costs nothing, and is a fair *relative* signal between two models
  on identical tickets.
- **LLM judge (optional, off by default)**: `scripts/run_eval.py --judge` adds a blind
  Claude judge with a fixed anchored 1–5 rubric via structured outputs. It needs a paid
  API key, so it is never required and no number in this README depends on it.
- **Cost**: both models are 8B on identical hardware, so $/ticket = latency × GPU rate.
- **MMLU-lite**: 24 bundled multiple-choice questions — a smoke alarm for gross capability
  regressions (format collapse, catastrophic forgetting), not a benchmark.

## CI

Lint (ruff) → unit tests (covering the eval harness, priority policy, data prep,
API) → offline data-prep + training-config smoke tests → Docker build + `/triage`
smoke test. Full training never runs in CI. All infra is free-tier: GitHub Actions
(public repo), HF Spaces (CPU showcase mode), local GPU training.

## Dataset & license

Training data derives from [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(CDLA-Sharing-1.0). Code is MIT.
