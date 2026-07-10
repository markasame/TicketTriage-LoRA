---
title: TicketTriage-LoRA
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# TicketTriage-LoRA demo

Side-by-side comparison of base Qwen3 8B vs a QLoRA+DoRA fine-tune
for support-ticket triage (intent, priority, draft reply), with eval scores on
the page.

To deploy this Space:

1. Copy `app/app.py` → `app.py`, `app/requirements.txt` → `requirements.txt`
2. Copy `src/tickettriage/` and `results/eval_results.json` preserving paths
3. Push to a Gradio Space (free CPU tier is enough — showcase mode serves
   precomputed eval outputs; set `BASE_BACKEND`/`FT_BACKEND` env vars for live
   inference against an Ollama/vLLM endpoint)

Or run `scripts/deploy_space.py` from the repo root to do all of that.
