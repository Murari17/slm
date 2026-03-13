from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def get_pipeline_python() -> str:
    # Use the project venv if it exists so runs are consistent across machines.
    candidates = [
        BASE_DIR / ".venv" / "bin" / "python",  # macOS/Linux path
        BASE_DIR / ".venv" / "Scripts" / "python.exe",  # Windows path
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    python_exec = get_pipeline_python()
    print(f"\n=== Running: {script_name} ===")
    print(f"Using Python: {python_exec}")
    result = subprocess.run([python_exec, str(script_path)], cwd=str(BASE_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({script_name}) with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full text-to-model pipeline.")
    parser.add_argument("--epochs", type=int, default=12, help="CPU training epochs for train.py")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    args = parser.parse_args()

    # Build the text corpus and dataset first.
    run_step("textextraction.py")
    run_step("clean_text.py")
    run_step("sentence_split.py")
    run_step("create_dataset.py")

    if not args.skip_tokenizer:
        run_step("train_tokenizer.py")

    if not args.skip_train:
        run_step("train.py", extra_env={"EPOCHS": str(args.epochs)})

    print("\nPipeline complete.")
    print("Run: python generate.py")


if __name__ == "__main__":
    main()
