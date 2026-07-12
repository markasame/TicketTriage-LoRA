---
title: TicketTriage-LoRA
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
---

# TicketTriage-LoRA demo (static showcase)

Side-by-side comparison of base Qwen3 8B vs a QLoRA fine-tune for
support-ticket triage, serving the *actual* precomputed outputs from the
before/after eval on 216 held-out tickets (`eval_results.json`).

Static because HF now requires a PRO subscription for Gradio/Docker Spaces on
free hardware — this page keeps the demo 100% free. The Gradio app
(`app/app.py` in the repo) still works locally and supports live inference.

Deployed with `scripts/deploy_space.py` from
[github.com/markasame/TicketTriage-LoRA](https://github.com/markasame/TicketTriage-LoRA).
