#!/usr/bin/env python3
"""
Figure 2 (combined):
- LEFT  (a): Offline training with a single "fair" metric proxy across 4 models.
- RIGHT (b): Material removed in 10 minutes.

Panel (a) fair-metric rule used here (based on available df columns only):
- ACT: use (val/loss)^2 to map L1+KL scale closer to MSE-like scale.
- pi0.5, SmolVLA, Diffusion: keep val/loss as-is (already MSE-style losses).

Usage:
1) In Colab with an existing `df` variable:
   %run -i figures.py

2) From CSV:
   python figures.py --csv path/to/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CONSISTENT COLORS ACROSS PANELS
# ==============================================================================
MODEL_COLORS = {
    "ACT": "#d62728",
    "pi0.5": "#ff7f0e",
    "SmolVLA": "#1f77b4",
    "Diffusion": "#2ca02c",
    "Teleop": "#2E7D32",
}

# ==============================================================================
# (b) MATERIAL REMOVED IN 10 MINUTES
# ==============================================================================
TOTAL_GRAMS_TELEOP = 2542
TOTAL_RECORDED_SECONDS = 2459.4  # ~40.99 min

teleop_g_per_min = TOTAL_GRAMS_TELEOP / (TOTAL_RECORDED_SECONDS / 60.0)
teleop_g_10min = teleop_g_per_min * 10.0

g_10min = {
    "pi0.5": 404,
    "SmolVLA": 113,
    "ACT": 169,
    "Diffusion": 57,
}

models_perf = ["Teleop", "pi0.5", "SmolVLA", "ACT", "Diffusion"]
means_10min = [
    teleop_g_10min,
    g_10min["pi0.5"],
    g_10min["SmolVLA"],
    g_10min["ACT"],
    g_10min["Diffusion"],
]
colors_bar = [MODEL_COLORS[m] for m in models_perf]

# ==============================================================================
# (a) OFFLINE TRAINING (FROM df)
# df must contain:
#   Step,
#   diffpol - val/loss, smolvla - val/loss, pi05 - val/loss, act - val/loss
# ==============================================================================
SERIES = [
    ("act - val/loss", "ACT"),
    ("pi05 - val/loss", "pi0.5"),
    ("smolvla - val/loss", "SmolVLA"),
    ("diffpol - val/loss", "Diffusion"),
]


def fair_loss_transform(label: str, y: np.ndarray) -> np.ndarray:
    """Single comparison scale for panel (a), using only available columns."""
    if label == "ACT":
        return np.square(y)  # ACT is L1+KL; square to approximate MSE-like scale.
    return y


def build_figure(df: pd.DataFrame, out_path: Path) -> None:
    required = ["Step"] + [col for col, _ in SERIES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- LEFT: offline training curves (a) with fair metric ----
    best_rows = []
    max_left_y = 0.0

    for col, label in SERIES:
        color = MODEL_COLORS[label]
        sub = df[["Step", col]].dropna().sort_values("Step")
        if sub.empty:
            continue

        x = sub["Step"].to_numpy()
        y_raw = sub[col].to_numpy(dtype=float)
        y_fair = fair_loss_transform(label, y_raw)
        max_left_y = max(max_left_y, float(np.max(y_fair)))

        ax_left.plot(
            x,
            y_fair,
            marker="o",
            linewidth=2.7,
            markersize=5.5,
            label=label,
            color=color,
            alpha=0.95,
        )

        best_i = int(np.argmin(y_fair))
        best_step = int(x[best_i])
        best_loss_fair = float(y_fair[best_i])
        best_loss_raw = float(y_raw[best_i])
        best_rows.append((label, best_step, best_loss_fair, best_loss_raw))

        ax_left.plot(
            best_step,
            best_loss_fair,
            marker="*",
            markersize=13,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1.4,
            zorder=10,
        )

        ax_left.annotate(
            f"{best_loss_fair:.4f}\n@{best_step:,}",
            xy=(best_step, best_loss_fair),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6),
            zorder=11,
        )

    ax_left.set_title("(a) Offline Training", fontsize=14, fontweight="bold", pad=10)
    ax_left.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax_left.set_ylabel("Validation Loss (single fair scale)", fontsize=12, fontweight="bold")
    ax_left.grid(True, alpha=0.3, linestyle="--")
    ax_left.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax_left.set_xlim(0, 30000)
    ax_left.set_ylim(0, max(0.1, max_left_y * 1.15))

    ax_left.text(
        0.02,
        0.98,
        "★ = best val loss\nACT transformed as (val/loss)^2",
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
    )

    # ---- RIGHT: material removed bars (b) ----
    bars = ax_right.bar(models_perf, means_10min, color=colors_bar, alpha=0.85)

    ax_right.set_title("(b) Material Removed (10 min)", fontsize=14, fontweight="bold", pad=10)
    ax_right.set_xlabel("Policy / Operator", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Removed Material (g / 10 min)", fontsize=12, fontweight="bold")
    ax_right.grid(axis="y", alpha=0.3, linestyle="--")

    ymax = max(means_10min) * 1.15
    ax_right.set_ylim(0, ymax)

    for b, v in zip(bars, means_10min):
        ax_right.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.02,
            f"{v:.0f} g",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_right.text(
        0.02,
        0.02,
        (
            f"Teleop mean: {teleop_g_per_min:.1f} g/min from "
            f"{TOTAL_GRAMS_TELEOP} g / {TOTAL_RECORDED_SECONDS/60:.2f} min"
        ),
        transform=ax_right.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", out_path)
    print(f"Teleop computed: {teleop_g_per_min:.2f} g/min -> {teleop_g_10min:.1f} g / 10 min")
    print("\nBest checkpoints (panel a, fair metric):")
    for label, step, fair_loss, raw_loss in best_rows:
        print(f"  {label:10s} best @ step {step:,}  fair={fair_loss:.6f}  raw={raw_loss:.6f}")


def _resolve_df(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)

    # For notebook use with `%run -i figures.py`
    if "df" in globals():
        df_obj = globals()["df"]
        if isinstance(df_obj, pd.DataFrame):
            return df_obj
    raise RuntimeError(
        "No dataframe found. Use `%run -i figures.py` with `df` already loaded, "
        "or run `python figures.py --csv your_metrics.csv`."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="CSV with Step and val/loss columns.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUTDIR / "figure2_combined_fair_a_b.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    df_in = _resolve_df(args.csv)
    build_figure(df_in, Path(args.out))
