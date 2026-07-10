#!/usr/bin/env bash
# One-shot pipeline for a RunPod RTX 4090 pod (image: runpod/pytorch:2.5+ cuda12.x).
# Total cost at $0.44/hr: data prep + training + eval + export typically < $2.
#
# Prereqs (set in the pod):
#   export HF_TOKEN=hf_...            # gated Llama 3.1 access
#   export ANTHROPIC_API_KEY=sk-...   # for the LLM judge (optional: --skip-judge)
#
# Usage: bash scripts/runpod_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/5 environment ==="
pip install -q -e ".[eval]"
pip install -q "unsloth" "trl>=0.12" "peft>=0.13" "datasets" "bitsandbytes" || \
  pip install -q "trl>=0.12" "peft>=0.13" "datasets" "bitsandbytes" accelerate
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"

echo "=== 2/5 data ==="
python scripts/prepare_data.py

echo "=== 3/5 train (QLoRA + DoRA) ==="
python scripts/train.py

echo "=== 4/5 eval (base vs fine-tuned) ==="
python scripts/run_eval.py \
  --base "hf:meta-llama/Llama-3.1-8B-Instruct" \
  --finetuned "hf:meta-llama/Llama-3.1-8B-Instruct@models/tickettriage-lora" \
  ${ANTHROPIC_API_KEY:+} $( [ -z "${ANTHROPIC_API_KEY:-}" ] && echo --skip-judge )

echo "=== 5/5 export GGUF ==="
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF >/dev/null
  cmake --build llama.cpp/build --target llama-quantize -j >/dev/null
fi
python scripts/export_gguf.py

echo "=== done ==="
echo "Download before stopping the pod:"
echo "  models/tickettriage-lora/   (adapter + training_log.json)"
echo "  models/gguf/tickettriage-q4_k_m.gguf"
echo "  results/eval_results.json, results/eval_report.md"
