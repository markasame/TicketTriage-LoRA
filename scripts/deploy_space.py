"""Deploy the demo to Hugging Face Spaces.

Requires: huggingface_hub login (`hf auth login` or HF_TOKEN env var) and a
completed eval (results/eval_results.json) so the Space has showcase data.

  python scripts/deploy_space.py --space <username>/tickettriage-lora

Default is a **static** Space (app/static/): HF requires a PRO subscription to
host Gradio/Docker Spaces even on free cpu-basic hardware, and this project is
100% free by policy. The static page serves the same precomputed showcase the
Gradio app would in showcase mode. Pass --sdk gradio to deploy the Gradio app
instead if your account can host it.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent


def stage_static(stage: Path, results: Path) -> None:
    shutil.copy(ROOT / "app" / "static" / "index.html", stage / "index.html")
    shutil.copy(ROOT / "app" / "static" / "README.md", stage / "README.md")
    shutil.copy(results, stage / "eval_results.json")


def stage_gradio(stage: Path, results: Path) -> None:
    shutil.copy(ROOT / "app" / "app.py", stage / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", stage / "requirements.txt")
    shutil.copy(ROOT / "app" / "README.md", stage / "README.md")
    shutil.copytree(ROOT / "src", stage / "src")
    (stage / "results").mkdir()
    shutil.copy(results, stage / "results" / "eval_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="e.g. myuser/tickettriage-lora")
    parser.add_argument("--sdk", choices=["static", "gradio"], default="static",
                        help="static is free for everyone; gradio needs HF PRO")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    results = ROOT / "results" / "eval_results.json"
    if not results.exists():
        raise SystemExit("results/eval_results.json missing — run scripts/run_eval.py first "
                         "(the Space serves precomputed eval outputs in showcase mode)")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk=args.sdk, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage_static if args.sdk == "static" else stage_gradio)(stage, results)
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space")

    print(f"deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
