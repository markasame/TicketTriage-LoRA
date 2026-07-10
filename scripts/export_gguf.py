"""Merge the LoRA adapter into the base model and export GGUF (q4_k_m) for llama.cpp/Ollama.

Run on the training box (needs the merged fp16 model in RAM, ~16GB):
  python scripts/export_gguf.py --adapter models/tickettriage-lora --out models/gguf

Then serve locally:
  ollama create tickettriage -f models/gguf/Modelfile
  ollama run tickettriage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_SYSTEM = (
    "You are TicketTriage, an assistant for a customer support team. "
    "You classify support tickets, score their priority, and draft replies. "
    "Follow the task instruction exactly and answer in the requested format."
)
MODELFILE = (
    "FROM ./tickettriage-q4_k_m.gguf\n"
    "PARAMETER temperature 0.2\n"
    "PARAMETER num_ctx 4096\n"
    f'SYSTEM """{_SYSTEM}"""\n'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/tickettriage-lora")
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", default="models/gguf")
    parser.add_argument("--llama-cpp", default="llama.cpp", help="path to a llama.cpp checkout")
    args = parser.parse_args()

    out = Path(args.out)
    merged_dir = out / "merged"
    out.mkdir(parents=True, exist_ok=True)

    # 1. merge adapter -> fp16 model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base {args.base} (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging adapter...")
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(merged_dir)

    # 2. convert to GGUF + quantize q4_k_m via llama.cpp
    convert = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not convert.exists():
        sys.exit(
            f"llama.cpp not found at {args.llama_cpp}. Clone it first:\n"
            "  git clone --depth 1 https://github.com/ggml-org/llama.cpp"
        )
    f16_gguf = out / "tickettriage-f16.gguf"
    q4_gguf = out / "tickettriage-q4_k_m.gguf"
    subprocess.run(
        [sys.executable, str(convert), str(merged_dir), "--outfile", str(f16_gguf),
         "--outtype", "f16"],
        check=True,
    )
    quantize = Path(args.llama_cpp) / "build" / "bin" / "llama-quantize"
    if not quantize.exists():
        quantize = Path(args.llama_cpp) / "llama-quantize"  # prebuilt release layout
    subprocess.run([str(quantize), str(f16_gguf), str(q4_gguf), "q4_k_m"], check=True)

    (out / "Modelfile").write_text(MODELFILE, encoding="utf-8")
    print(f"done: {q4_gguf}\nOllama: ollama create tickettriage -f {out / 'Modelfile'}")


if __name__ == "__main__":
    main()
