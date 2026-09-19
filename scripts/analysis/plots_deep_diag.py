#!/usr/bin/env python3
"""深挖诊断可视化：机制图 / 轨迹图 / 几何图（读 outputs/deep_diag/*.json）。

输出 (figures/deep_diag/):
  figM_mechanism.{png,pdf}   四象限分布 + NQ 救回率 vs QN 保护率 + label-free 拖拽曲线
  figT_trajectory.{png,pdf}  prefix margin 轨迹（early vs full-correct vs full-wrong）
  figG_geometry.{png,pdf}    cos(f_s, gt) 与几何 margin 随尺度
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "outputs" / "deep_diag"
OUT = ROOT / "figures" / "deep_diag"
OUT.mkdir(parents=True, exist_ok=True)

SETTINGS = [("PET", "text"), ("PET", "gen"), ("EUROSAT", "text"),
            ("EUROSAT", "gen"), ("FLO", "text"), ("FLO", "gen")]
LBL = {"PET": "Pets", "EUROSAT": "EuroSAT", "FLO": "Flowers"}
COL = {"PET": "#1f77b4", "EUROSAT": "#d62728", "FLO": "#2ca02c"}


def load(ds, proto):
    return json.loads((DIAG / f"{ds}_ViT-B32_{proto}.json").read_text())


def fig_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    names = [f"{LBL[ds]}/{p}" for ds, p in SETTINGS]
    x = np.arange(len(SETTINGS))

    # (a) 四象限占比堆叠
    ax = axes[0]
    quad_colors = {"QQ": "#bdbdbd", "QN": "#d62728", "NQ": "#2ca02c", "NN": "#636363"}
    bottoms = np.zeros(len(SETTINGS))
    for q in ("QQ", "NQ", "QN", "NN"):
        vals = []
        for ds, p in SETTINGS:
            d = load(ds, p)
            vals.append(d["quadrants"][q].get("frac", 0) * 100)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=quad_colors[q], width=0.62,
               label=q, edgecolor="w", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylabel("share of samples (%)", fontsize=8)
    ax.set_title("(a) s1 × s10 correctness quadrants\n"
                 "Q=correct, N=wrong (order: s1,s10)", fontsize=8.5)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax.tick_params(labelsize=7.5)

    # (b) NQ rescue vs QN protect（融合行为）
    ax = axes[1]
    nq = [load(ds, p)["quadrants"]["NQ"]["fused_acc"] * 100 for ds, p in SETTINGS]
    qn = [load(ds, p)["quadrants"]["QN"]["fused_acc"] * 100 for ds, p in SETTINGS]
    w = 0.36
    ax.bar(x - w / 2, nq, width=w, color="#2ca02c", label="NQ rescue (fused acc | s1✗,s10✓)")
    ax.bar(x + w / 2, qn, width=w, color="#d62728", label="QN protect (fused acc | s1✓,s10✗)")
    for xi, (a, b) in enumerate(zip(nq, qn)):
        ax.text(xi - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylabel("fused accuracy (%)", fontsize=8)
    ax.set_title("(b) Fusion behavior on conflict samples:\n"
                 "EUROSAT fusion rescues NQ far less", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=7.5)

    # (c) label-free drag 曲线
    ax = axes[2]
    for ds, p in SETTINGS:
        d = load(ds, p)["drag"]
        ts = np.arange(1, 11)
        ls = "-" if p == "text" else "--"
        ax.plot(ts, np.array(d["to_s10_curve"]) * 100, ls, color=COL[ds],
                lw=1.5, marker="o", ms=2.5, alpha=0.9)
        ax.plot(ts, np.array(d["to_s1_curve"]) * 100, ls, color=COL[ds],
                lw=0.9, alpha=0.45)
    ax.set_xlabel("prefix length t", fontsize=8)
    ax.set_ylabel("% of s1≠s10 samples", fontsize=8)
    ax.set_title("(c) label-free: prefix fused agrees with\n"
                 "s10 (bold) / s1 (faint). EUROSAT/gen: dragged down", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, lw=0.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figM_mechanism.{ext}", dpi=170)
    print("[saved] figM_mechanism")


def fig_trajectory():
    sets4 = [("PET", "text"), ("PET", "gen"), ("FLO", "text"), ("FLO", "gen")]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4), sharex=True)
    for j, (ds, p) in enumerate(sets4):
        d = load(ds, p)["traj_groups"]
        ts = np.arange(1, d[list(d)[0]]["n"] * 0 + 11) if d else np.arange(1, 11)
        ts = np.arange(1, 11)
        for row, key, ylab in ((0, "M_mean", "prefix margin"), (1, "H_mean", "prefix entropy")):
            ax = axes[row, j]
            for grp, c, lab in (("early", "#2ca02c", "early-exit"),
                                ("full_correct", "#1f77b4", "full (correct)"),
                                ("full_wrong", "#d62728", "full (wrong)")):
                if grp not in d:
                    continue
                m = np.array(d[grp][key])
                s = np.array(d[grp][key.replace("_mean", "_sem")])
                ax.plot(ts, m, color=c, lw=1.6, label=f"{lab} (n={d[grp]['n']})")
                ax.fill_between(ts, m - s, m + s, color=c, alpha=0.18, lw=0)
            if row == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper left")
            ax.set_title(f"{LBL[ds]}/{p}", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.4)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
    for j in range(4):
        axes[1, j].set_xlabel("prefix length t", fontsize=8)
    fig.suptitle("Prefix dynamics by execution group: early exits stabilize fast; "
                 "full runs stay low-margin (wrong ones lowest)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figT_trajectory.{ext}", dpi=170)
    print("[saved] figT_trajectory")


def fig_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ts = np.arange(1, 11)
    for ds, p in SETTINGS:
        d = load(ds, p)["geometry"]
        ls = "-" if p == "text" else "--"
        axes[0].plot(ts, d["cos_gt_mean_by_scale"], ls, color=COL[ds], lw=1.6,
                     marker="o", ms=3, label=f"{LBL[ds]}/{p}")
        axes[1].plot(ts, d["geo_margin_mean_by_scale"], ls, color=COL[ds], lw=1.6,
                     marker="o", ms=3)
    axes[0].set_ylabel(r"mean $\cos(f_s,\, p_{gt})$", fontsize=8.5)
    axes[1].set_ylabel(r"mean geometric margin", fontsize=8.5)
    for ax, ti in zip(axes, ("(a) alignment with GT prototype",
                             "(b) cos to GT − cos to best other")):
        ax.set_xlabel("scale s", fontsize=8.5)
        ax.set_title(ti, fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7.5)
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figG_geometry.{ext}", dpi=170)
    print("[saved] figG_geometry")


def fig_mechanism_compact():
    """Paper-ready 2-panel 微图 v4: (a) quadrants, (b) NQ rescue。

    修复 v3: 图例移出坐标区（不再压标题/柱体）、缩短轴标签避免溢出、
    放大字号，TrueType 嵌入；设计尺寸≈显示尺寸（0.92 栏宽）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(3.12, 1.28))
    names = [f"{LBL[ds]}/{p}" for ds, p in SETTINGS]
    x = np.arange(len(SETTINGS))

    ax = axes[0]
    quad_colors = {"QQ": "#c9c9c9", "QN": "#d62728", "NQ": "#2ca02c", "NN": "#5f5f5f"}
    bottoms = np.zeros(len(SETTINGS))
    for q in ("NQ", "QQ", "QN", "NN"):
        vals = np.array([load(ds, p)["quadrants"][q].get("frac", 0) * 100
                         for ds, p in SETTINGS])
        ax.bar(x, vals, bottom=bottoms, color=quad_colors[q], width=0.62,
               label=q, edgecolor="w", linewidth=0.3)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.2)
    ax.set_ylabel("share (%)", fontsize=5.8)
    ax.set_title(r"(a) $s_1{\times}s_{10}$ quadrants", fontsize=6.2, pad=12)
    ax.legend(fontsize=5.0, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              frameon=False, handlelength=1.1, columnspacing=0.9,
              handletextpad=0.35, borderpad=0.0, borderaxespad=0.0)
    ax.tick_params(labelsize=5.2, pad=1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    ax = axes[1]
    nq = [load(ds, p)["quadrants"]["NQ"]["fused_acc"] * 100 for ds, p in SETTINGS]
    ax.bar(x, nq, width=0.62, color=["#d62728" if ds == "EUROSAT" else "#2ca02c"
                                     for ds, p in SETTINGS])
    for xi, a in enumerate(nq):
        ax.text(xi, a + 1.5, f"{a:.0f}", ha="center", fontsize=5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.2)
    ax.set_ylabel("rescue (%)", fontsize=5.8)
    ax.set_ylim(0, 112)
    ax.set_title("(b) fusion rescue on NQ\n($s_1$ wrong, $s_{10}$ correct)",
                 fontsize=6.2, pad=12)
    ax.tick_params(labelsize=5.2, pad=1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.35)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    fig.tight_layout(pad=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figM_compact.{ext}", dpi=400, bbox_inches="tight")
    print("[saved] figM_compact")


if __name__ == "__main__":
    fig_mechanism()
    fig_mechanism_compact()
    fig_trajectory()
    fig_geometry()
