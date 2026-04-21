"""Render the architecture diagram as a PNG.

Run from the repo root:

    python docs/diagrams/_make_architecture.py

This is the only place that produces ``architecture.png``; checked-in
diagram images can be regenerated deterministically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

CHARCOAL = "#22272e"
GREY = "#9aa0a6"
ACCENT = "#1f6feb"
LIGHT = "#eef2f7"


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    ax.set_xlim(-8, 108)   # extra margins so side labels aren't clipped
    ax.set_ylim(0, 62)
    ax.axis("off")

    # Supervisor box (top)
    sup = patches.FancyBboxPatch(
        (35, 50),
        30,
        9,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        linewidth=1.2,
        edgecolor=CHARCOAL,
        facecolor=LIGHT,
    )
    ax.add_patch(sup)
    ax.text(
        50,
        54.5,
        "Supervisor",
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=12,
        fontweight="bold",
    )

    # Workers row
    worker_x = [10, 40, 70]
    worker_names = ["Worker A", "Worker B", "Worker C"]
    for x, name in zip(worker_x, worker_names, strict=True):
        box = patches.FancyBboxPatch(
            (x, 27),
            20,
            8,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            linewidth=1.2,
            edgecolor=CHARCOAL,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 10, 31, name, ha="center", va="center", color=CHARCOAL, fontsize=11)

    # Shared context (bottom)
    sc = patches.FancyBboxPatch(
        (10, 5),
        80,
        9,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        linewidth=1.4,
        edgecolor=ACCENT,
        facecolor=LIGHT,
    )
    ax.add_patch(sc)
    ax.text(
        50,
        9.5,
        "SharedContext  →  Backend (Local | custom)",
        ha="center",
        va="center",
        color=ACCENT,
        fontsize=11,
        fontweight="bold",
    )

    # Supervisor → workers (decompose / fan-out)
    for x in worker_x:
        ax.annotate(
            "",
            xy=(x + 10, 35),
            xytext=(50, 50),
            arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=1.0),
        )
    # Label sits to the right of the rightmost arrow, clear of all boxes
    ax.text(74, 44, "decompose +\nfan-out", color=CHARCOAL, fontsize=8, style="italic")

    # Workers ↔ SharedContext (read/write) — bidirectional
    for x in worker_x:
        ax.annotate(
            "",
            xy=(x + 10, 14),
            xytext=(x + 10, 27),
            arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2),
        )
    # Label in the right margin, between the worker row and SharedContext
    ax.text(
        92,
        19,
        "read/write\nshared context",
        color=ACCENT,
        fontsize=8,
        style="italic",
        ha="left",
    )

    # SharedContext → Supervisor (synthesis) — curved LEFT to avoid workers
    ax.annotate(
        "",
        xy=(38, 50),
        xytext=(10, 14),
        arrowprops=dict(
            arrowstyle="->",
            color=GREY,
            lw=0.9,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.35",
        ),
    )
    # Label in the left margin
    ax.text(
        -7,
        32,
        "synthesize\n(reads full\ntrace)",
        color=GREY,
        fontsize=8,
        style="italic",
        ha="left",
    )

    out = Path(__file__).parent / "architecture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
