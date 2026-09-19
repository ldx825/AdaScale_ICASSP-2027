#!/usr/bin/env python3
"""AdaScale 总体架构图（单栏版，ICASSP）。

竖向流程：
  test image → scale crops → Level-1 unlabeled fusion-utility diagnosis
   ├─ SINGLE: largest scale only (1 forward)
   └─ FUSED : Level-2 reliability stopping loop → probability-level soft fusion
输出: figures/architecture/fig_arch.{png,pdf} + paper/figures/architecture/（副本）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams

rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "architecture"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#dce9f7"
GREEN = "#dcefd8"
ORANGE = "#fdebd3"
GRAY = "#f0f0f0"
EDGE = "#333333"


def box(ax, x, y, w, h, text, fc, fs=7.4, weight="normal", style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=EDGE, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, weight=weight, linespacing=1.35)


def arrow(ax, p1, p2, style="-|>", lw=0.9, color=EDGE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, lw=lw, color=color,
                                 mutation_scale=7, zorder=1,
                                 shrinkA=0.5, shrinkB=0.5))


def main():
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.2); ax.axis("off")

    # ---- 输入 ----
    box(ax, 0.15, 6.9, 2.2, 1.0, "test image $x$", GRAY, fs=7.4)
    box(ax, 3.0, 6.9, 3.4, 1.0, "scale crops\n$s_1,\\dots,s_{10}$ (RRC)", GRAY, fs=7.4)
    arrow(ax, (2.35, 7.4), (3.0, 7.4))
    # 小尺度示意：3 个大小不同的框
    for i, (w, x) in enumerate([(0.22, 6.75), (0.34, 7.15), (0.5, 7.66)]):
        ax.add_patch(Rectangle((x, 7.17), w, w, fc="#c9d6ea", ec=EDGE, lw=0.6, zorder=3))
    arrow(ax, (6.4, 7.4), (7.2, 7.4))

    # ---- Level 1 ----
    box(ax, 7.2, 6.55, 2.65, 1.7,
        "Level 1: fusion-utility\ndiagnosis on unlabeled\n$\\mathcal{U}$:\n"
        "$\\Delta H = \\overline{H}(p^{\\,fused})-\\overline{H}(p^{\\,single})$",
        BLUE, fs=7.0)
    arrow(ax, (8.5, 6.55), (8.5, 5.6))

    # 菱形判断
    ax.add_patch(plt.Polygon([[8.5, 5.55], [9.55, 4.95], [8.5, 4.35], [7.45, 4.95]],
                             fc=ORANGE, ec=EDGE, lw=0.8, zorder=2))
    ax.text(8.5, 4.95, "$\\Delta H > \\theta$ ?", ha="center", va="center",
            fontsize=7.0, zorder=3)

    # SINGLE 分支（左）
    box(ax, 0.5, 4.25, 3.3, 1.1,
        "SINGLE: largest scale only\n(1 forward)", GREEN, fs=7.0)
    arrow(ax, (7.45, 4.95), (3.8, 4.8))
    ax.text(5.4, 5.05, "yes", fontsize=6.8, style="italic")

    # FUSED 分支（右）
    arrow(ax, (8.5, 4.35), (8.5, 3.55))
    ax.text(8.75, 3.9, "no", fontsize=6.8, style="italic")
    box(ax, 5.4, 2.05, 4.45, 1.5,
        "Level 2: scale loop $t=1..10$\nencode $s_t$; exit once\n"
        "agree$_r$ $\\wedge$ margin $\\geq\\delta$ $\\wedge$ JS $\\leq\\varepsilon$",
        BLUE, fs=7.0)
    # 自环箭头示意迭代
    ax.add_patch(FancyArrowPatch((5.75, 2.05), (4.9, 1.45),
                                 connectionstyle="arc3,rad=0.55",
                                 arrowstyle="-|>", lw=0.8, color=EDGE,
                                 mutation_scale=7, zorder=1))
    ax.text(4.62, 1.72, "$t{+}1$", fontsize=6.6, style="italic")

    # ---- 输出 ----
    arrow(ax, (7.6, 2.05), (7.6, 1.35))
    arrow(ax, (2.15, 4.25), (2.15, 1.35))
    box(ax, 1.2, 0.15, 7.6, 1.15,
        "output: probability-level soft fusion\n"
        "$p_{out} \\propto \\sum_{s\\in\\mathcal{V}} w_s\\,\\mathrm{softmax}(f_s^{\\top} P / \\tau_s)$"
        "\u2003 ($w_s{=}s$, $\tau_s{=}0.02$; single view: unchanged)",
        GRAY, fs=6.9)
    # SINGLE 直连输出
    arrow(ax, (2.15, 4.25), (2.15, 1.35))

    fig.tight_layout(pad=0.1)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_arch.{ext}", dpi=220)
    print("[saved]", OUT / "fig_arch.pdf")


if __name__ == "__main__":
    main()
