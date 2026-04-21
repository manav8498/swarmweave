"""Render the benchmark results bar chart.

Restrained palette: deep charcoal background-text on a clean white field with
one accent color for the shared-context condition. No gradients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHARCOAL = "#22272e"
GREY = "#9aa0a6"
ACCENT = "#1f6feb"


def render_chart(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iso = summary["isolated"]
    sc = summary["shared"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=160)
    fig.suptitle(
        "swarmweave benchmark — isolated vs. shared context",
        color=CHARCOAL,
        fontsize=12,
        fontweight="bold",
    )

    # Accuracy
    ax = axes[0]
    bars = ax.bar(
        ["isolated", "shared"],
        [iso["accuracy"], sc["accuracy"]],
        color=[GREY, ACCENT],
        width=0.55,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy (pass rate)", color=CHARCOAL)
    ax.set_title("accuracy", color=CHARCOAL, fontsize=11)
    for bar, val in zip(bars, [iso["accuracy"], sc["accuracy"]], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            color=CHARCOAL,
            fontsize=10,
        )

    # Tokens
    ax2 = axes[1]
    iso_tokens = iso["input_tokens"] + iso["output_tokens"]
    sc_tokens = sc["input_tokens"] + sc["output_tokens"]
    bars2 = ax2.bar(
        ["isolated", "shared"],
        [iso_tokens, sc_tokens],
        color=[GREY, ACCENT],
        width=0.55,
    )
    ax2.set_ylabel("total tokens (input + output)", color=CHARCOAL)
    ax2.set_title("token usage", color=CHARCOAL, fontsize=11)
    for bar, val in zip(bars2, [iso_tokens, sc_tokens], strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:,}",
            ha="center",
            color=CHARCOAL,
            fontsize=10,
        )

    for ax_ in axes:
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)
        ax_.tick_params(colors=CHARCOAL)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
