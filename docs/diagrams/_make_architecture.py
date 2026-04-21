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
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Supervisor box (top)
    sup = patches.FancyBboxPatch(
        (35, 47),
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
        51.5,
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
            (x, 24),
            20,
            8,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            linewidth=1.2,
            edgecolor=CHARCOAL,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 10, 28, name, ha="center", va="center", color=CHARCOAL, fontsize=11)

    # Shared context (bottom)
    sc = patches.FancyBboxPatch(
        (10, 4),
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
        9,
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
            xy=(x + 10, 32.5),
            xytext=(50, 47),
            arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=1.0),
        )
    ax.text(72, 40, "decompose +\nfan-out", color=CHARCOAL, fontsize=8, style="italic")

    # Workers ↔ shared context (read/write)
    for x in worker_x:
        ax.annotate(
            "",
            xy=(x + 10, 13),
            xytext=(x + 10, 24),
            arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2),
        )
    ax.text(73, 18, "read/write\nshared context", color=ACCENT, fontsize=8, style="italic")

    # Shared context → supervisor (synthesis input)
    ax.annotate(
        "",
        xy=(50, 47),
        xytext=(50, 13),
        arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9, linestyle="dashed"),
    )
    ax.text(52, 30, "synthesize\n(reads full trace)", color=GREY, fontsize=8, style="italic")

    out = Path(__file__).parent / "architecture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
