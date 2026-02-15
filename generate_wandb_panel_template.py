#!/usr/bin/env python
"""
Generate a reusable panel template for Weights & Biases benchmark dashboards.

This script does not call private WandB workspace APIs. It emits a JSON template
with the recommended panel metrics/regex so you can quickly recreate the same
layout in the W&B UI (or keep it as paper metadata).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_template(project: str, entity: str | None) -> dict:
    base = {
        "project": project,
        "entity": entity,
        "sections": [
            {
                "title": "Core",
                "panels": [
                    {"title": "Train Loss", "metric_regex": "^train/loss$"},
                    {"title": "Val Loss", "metric_regex": "^val/loss$"},
                    {"title": "Learning Rate", "metric_regex": "^train/lr$"},
                    {"title": "Grad Norm", "metric_regex": "^train/grad_norm$"},
                    {"title": "Epochs", "metric_regex": "^train/epochs$"},
                    {"title": "Best Val Loss", "metric_regex": "^val/best_val_loss$"},
                    {"title": "Best Val Step", "metric_regex": "^val/best_val_step$"},
                ],
            },
            {
                "title": "Model Specific",
                "panels": [
                    {"title": "ACT l1", "metric_regex": "^train/l1_loss$"},
                    {"title": "ACT kld", "metric_regex": "^train/kld_loss$"},
                    {
                        "title": "SmolVLA Loss Stages",
                        "metric_regex": "^train/losses_after_(forward|in_ep_bound|rm_padding)$",
                    },
                    {
                        "title": "pi05 Loss Per Dim",
                        "metric_regex": "^train/loss_per_dim_.*$",
                    },
                    {
                        "title": "pi05 Val Loss Per Dim",
                        "metric_regex": "^val/loss_per_dim_.*$",
                    },
                ],
            },
        ],
    }
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity/team (optional)")
    parser.add_argument(
        "--output",
        default="wandb_panel_template.json",
        help="Output JSON template path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_template(project=args.project, entity=args.entity)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote panel template: {output_path}")


if __name__ == "__main__":
    main()
