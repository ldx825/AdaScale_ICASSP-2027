"""
plot_pareto.py
--------------
绘制 Acc–Compute Pareto 曲线（论文核心图）：
  - Fixed-K 曲线（K=1..10）
  - Stability 策略散点（细网格）
  - full-10 参考线

用法:
  python scripts/experiments/plot_pareto.py --dataset PET --backbone ViT-B/32 --proto text
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
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "methods")

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

    sim = pd.read_csv(os.path.join(base, f"sim_results_{args.proto}.csv"))
    fine_path = os.path.join(base, f"fine_grid_{args.proto}.csv")
    fine = pd.read_csv(fine_path) if os.path.exists(fine_path) else None

    fx = sim[sim.kind == "fixed"].sort_values("avg_scales")
    st = sim[sim.kind == "stability"]
    mo = sim[sim.kind == "margin_only"]

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(fx.avg_scales, fx.acc * 100, "s-", color="tab:blue",
            label="Fixed-K (K=1..10)")
    if len(st):
        ax.scatter(st.avg_scales, st.acc * 100, s=14, alpha=0.55,
                   color="tab:orange", label="Stability policies (grid)")
    if fine is not None:
        ax.scatter(fine.avg_scales, fine.acc * 100, s=16, alpha=0.7,
                   color="tab:red", marker="^", label="Fine grid (r3/r4)")
    if len(mo):
        ax.scatter(mo.avg_scales, mo.acc * 100, s=20, alpha=0.7,
                   color="tab:green", marker="x", label="Margin-only")
    f10 = fx[fx.avg_scales == 10].acc.values[0] * 100
    ax.axhline(f10, color="gray", ls=":", lw=1)
    ax.annotate(f"full-10 = {f10:.2f}%", (6.2, f10 + 0.06), fontsize=8, color="gray")
    ax.set_xlabel("Average scales used (↓ compute)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{args.dataset} / {args.backbone} / proto={args.proto}")
    ax.legend(fontsize=8, loc="lower right")
    name = f"{args.dataset}_{tag}_{args.proto}_pareto"
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, name))


if __name__ == "__main__":
    main()
