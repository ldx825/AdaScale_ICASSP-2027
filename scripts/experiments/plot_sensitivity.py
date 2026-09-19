"""
plot_sensitivity.py
-------------------
超参敏感度曲线（文档 §15/E12）：
  - 对不同 (r, delta, eps) 组合画 accuracy vs avg_scales（曲线族）
  - 展示 AdaScale 的"预算-精度"可调性，避免只报单一 operating point

用法:
  python scripts/experiments/plot_sensitivity.py --dataset PET --backbone ViT-B/32 --proto text
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text")
    args = ap.parse_args()
    tag = args.backbone.replace("/", "")
    base = os.path.join(PROJECT_ROOT, "outputs", "methods", args.dataset, tag)
    fine = pd.read_csv(os.path.join(base, f"fine_grid_{args.proto}.csv"))
    sim = pd.read_csv(os.path.join(base, f"sim_results_{args.proto}.csv"))

    fine["r"] = fine["method"].str.extract(r"r(\d)")
    fine["delta"] = fine["method"].str.extract(r"d([0-9.]+)").astype(float)
    fine["eps"] = fine["method"].str.extract(r"e([0-9.]+)").astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    # 左：按 delta 分组（r3, eps=0.01）
    ax = axes[0]
    sub = fine[(fine.method.str.startswith("r3")) & (fine.eps == 0.01)]
    for d, g in sub.groupby("delta"):
        g = g.sort_values("avg_scales")
        ax.plot(g.avg_scales, g.acc * 100, "o-", label=f"$\\delta$={d}", ms=5)
    fx = sim[sim.kind == "fixed"].sort_values("avg_scales")
    ax.plot(fx.avg_scales, fx.acc * 100, "s--", color="gray", label="Fixed-K")
    ax.set_xlabel("avg scales"); ax.set_ylabel("acc (%)")
    ax.set_title(f"(a) effect of $\\delta$ (r=3, $\\epsilon$=0.01)")
    ax.legend(fontsize=8)

    # 右：按 eps 分组（r3, delta=0.25）
    ax = axes[1]
    sub = fine[(fine.method.str.startswith("r3")) & (fine.delta == 0.25)]
    for e, g in sub.groupby("eps"):
        g = g.sort_values("avg_scales")
        lbl = "no JS" if e == 1.0 else f"$\\epsilon$={e}"
        ax.plot(g.avg_scales, g.acc * 100, "o-", label=lbl, ms=5)
    ax.plot(fx.avg_scales, fx.acc * 100, "s--", color="gray", label="Fixed-K")
    ax.set_xlabel("avg scales"); ax.set_ylabel("acc (%)")
    ax.set_title(f"(b) effect of $\\epsilon$ (r=3, $\\delta$=0.25)")
    ax.legend(fontsize=8)

    fig.suptitle(f"{args.dataset} / {args.backbone} / {args.proto}", y=1.03)
    out_dir = os.path.join(PROJECT_ROOT, "figures", "methods")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.dataset}_{tag}_{args.proto}_sensitivity"
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
