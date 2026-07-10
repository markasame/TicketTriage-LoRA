"""Deploy the Gradio demo to Hugging Face Spaces.

Requires: huggingface_hub login (`hf auth login` or HF_TOKEN env var) and a
completed eval (results/eval_results.json) so the Space has showcase data.

  python scripts/deploy_space.py --space <username>/tickettriage-lora
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="e.g. myuser/tickettriage-lora")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    results = ROOT / "results" / "eval_results.json"
    if not results.exists():
        raise SystemExit("results/eval_results.json missing — run scripts/run_eval.py first "
                         "(the Space serves precomputed eval outputs in showcase mode)")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy(ROOT / "app" / "app.py", stage / "app.py")
        shutil.copy(ROOT / "app" / "requirements.txt", stage / "requirements.txt")
        shutil.copy(ROOT / "app" / "README.md", stage / "README.md")
        shutil.copytree(ROOT / "src", stage / "src")
        (stage / "results").mkdir()
        shutil.copy(results, stage / "results" / "eval_results.json")
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space")

    print(f"deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
