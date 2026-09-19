#!/usr/bin/env python3
"""Fig 2 (supplementary / extended version): Accuracy–compute Pareto across
methods: Full-10, Single-10, Fixed-K, B1 (margin stop), B2 (gating),
B3 (entropy filter), AdaScale v3.

x = average scale evaluations, y = accuracy. 2x3 subplots per setting.

用法: python scripts/paper/fig_pareto.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DATASETS = [("PET", "text"), ("PET", "gen"),
            ("EUROSAT", "text"), ("EUROSAT", "gen"),
            ("FLO", "text"), ("FLO", "gen")]
DS_NAME = {"PET": "Oxford-Pets", "EUROSAT": "EuroSAT", "FLO": "Flowers-102"}


def main():
    rows = {(r["dataset"], r["proto"]): r for r in csv.DictReader(
        open(ROOT / "outputs" / "final" / "all_results_v2.csv"))}

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for ax, (ds, proto) in zip(axes.ravel(), DATASETS):
        r = rows[(ds, proto)]
        pb = json.loads((ROOT / "outputs" / "prior_baselines" /
                         f"{ds}_ViT-B32_{proto}.json").read_text())
        m = pb["methods"]
        pts = []
        pts.append(("Full-10", 10.0, pb["full10_acc"], "s", "#444444"))
        pts.append(("Single-10", 1.0, pb["single10_independent_acc"], "^", "#888888"))
        for K in (5, 7):
            v = r.get(f"fixed{K}_acc")
            if v:
                pts.append((f"Fixed-{K}", float(K), float(v), "P", "#1f77b4"))
        b1 = max(m["margin_stop"], key=lambda x: x["acc"])
        pts.append(("Margin stop", b1["avg_scales"], b1["acc"], "X", "#d62728"))
        b2 = max(m["selective_gating"], key=lambda x: x["acc"])
        pts.append(("Gating", b2["avg_scales"], b2["acc"], "v", "#ff7f0e"))
        b3 = next(x for x in m["entropy_filter"] if "K=5" in str(x.get("param", "")))
        pts.append(("Entropy filt.", 10.0, b3["acc"], "*", "#9467bd"))
        v3 = float(r.get("v3_acc") or r["adascale_acc"])
        pts.append(("AdaScale", float(r["adascale_scales"]), v3, "D", "#2ca02c"))

        for name, x, y, mk, c in pts:
            ax.scatter(x, y, marker=mk, c=c, s=55, zorder=3,
                       edgecolors="k", linewidths=0.4)
        # 标注 AdaScale
        ax.annotate("AdaScale", (float(r["adascale_scales"]), v3),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5,
                    color="#2ca02c", fontweight="bold")
        ax.annotate("Full-10", (10.0, pb["full10_acc"]),
                    textcoords="offset points", xytext=(-32, -10), fontsize=7,
                    color="#444444")
        mode = r["mode"]
        ax.set_title(f"{DS_NAME[ds]} / {proto}  [{mode}]", fontsize=9)
        ax.set_xlabel("avg. scale evaluations", fontsize=8)
        ax.set_ylabel("accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(0.4, 10.8)
        ax.grid(alpha=0.25, linewidth=0.4)

    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#444444", label="Full-10"),
        plt.Line2D([], [], marker="^", ls="", color="#888888", label="Single-10"),
        plt.Line2D([], [], marker="P", ls="", color="#1f77b4", label="Fixed-K"),
        plt.Line2D([], [], marker="X", ls="", color="#d62728", label="Margin stop (AdapTTA-style)"),
        plt.Line2D([], [], marker="v", ls="", color="#ff7f0e", label="Gating (Selective-TTA-style)"),
        plt.Line2D([], [], marker="*", ls="", color="#9467bd", label="Entropy filter"),
        plt.Line2D([], [], marker="D", ls="", color="#2ca02c", label="AdaScale (ours)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy–compute Pareto: AdaScale vs fixed schedules and "
                 "prior-inspired adaptive baselines", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = ROOT / "paper" / "figures" / "figB_pareto"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=160)
    print(f"[saved] {out}.pdf/.png")


if __name__ == "__main__":
    main()
