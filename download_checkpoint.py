#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face Hub into this repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = REPO_ROOT / "outputs" / "hf_checkpoints"


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def resolve_output_dir(repo_id: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (DEFAULT_OUTPUT_BASE / sanitize_repo_id(repo_id)).resolve()


def download_checkpoint(
    model_repo: str,
    checkpoint: str,
    output_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    full_repo: bool = False,
) -> tuple[Path, Path]:
    local_dir = resolve_output_dir(model_repo, output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if not full_repo:
        allow_patterns = [f"checkpoints/{checkpoint}/pretrained_model/*"]

    snapshot_download(
        repo_id=model_repo,
        repo_type="model",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
    )

    policy_path = local_dir / "checkpoints" / checkpoint / "pretrained_model"
    return local_dir, policy_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download checkpoint files from a Hugging Face model repo."
    )
    parser.add_argument(
        "--model-repo",
        required=True,
        help="Model repo id, e.g. Autobrik/SO-ARM100-dump-pocket-cleaning2-smolvla",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint group to download. Default: best",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local destination directory. Default: outputs/hf_checkpoints/<owner__repo>",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/tag/commit revision.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, uses local huggingface-cli login token.",
    )
    parser.add_argument(
        "--full-repo",
        action="store_true",
        help="Download full repo instead of only checkpoints/<checkpoint>/pretrained_model/*",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    local_dir, policy_path = download_checkpoint(
        model_repo=args.model_repo,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        revision=args.revision,
        token=args.token,
        full_repo=args.full_repo,
    )

    print(f"Downloaded from: https://huggingface.co/{args.model_repo}")
    print(f"Local dir: {local_dir}")

    if args.full_repo:
        print("Full repository downloaded.")
        return

    if not policy_path.is_dir():
        raise FileNotFoundError(
            f"Expected policy directory was not downloaded: {policy_path}"
        )

    required_files = ("config.json", "model.safetensors")
    missing_files = [name for name in required_files if not (policy_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Downloaded directory is missing required files for inference: "
            + ", ".join(missing_files)
        )

    print(f"Policy path for run_inference_*: {policy_path}")


if __name__ == "__main__":
    main()
