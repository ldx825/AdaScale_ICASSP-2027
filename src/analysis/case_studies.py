"""
case_studies.py
---------------
案例研究：从 audit 数据中挑选典型样本并绘制预测轨迹（Fig for paper motivation）。

分组:
  - Stable Easy:  t_eq <= 2 且最终正确
  - Rescue:       最终正确，早期错误（首个正确 scale >= 3）
  - Harmful:      早期某 scale 正确，最终错误
  - Oscillatory:  翻转次数 >= 3

输出: figures/cases/<dataset>_<backbone>_<proto>_cases.{pdf,png}
      + cases_<...>.csv（样本索引明细，便于人工检查原图）

用法:
  python src/analysis/case_studies.py --dataset PET --backbone ViT-B/32 --proto gen
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="gen")
    ap.add_argument("--n_per_group", default=3, type=int)
    args = ap.parse_args()

    tag = args.backbone.replace("/", "")
    base = os.path.join(PROJECT_ROOT, "outputs", "scale_audit", args.dataset, tag)
    df = pd.read_parquet(os.path.join(base, f"per_sample_{args.proto}.parquet"))
    N = df.sample_id.nunique()
    T = df.scale_id.max()
    pv = df.pivot(index="sample_id", columns="scale_id", values="top1_class")
    preds = pv[np.arange(1, T + 1)].values
    gt = df[df.scale_id == 1].sort_values("sample_id")["ground_truth"].values
    m = df.pivot(index="sample_id", columns="scale_id", values="margin")[np.arange(1, T + 1)].values

    corr = preds == gt[:, None]
    final = preds[:, -1]
    flips = (preds[:, 1:] != preds[:, :-1]).sum(axis=1)

    # t_eq
    teq = np.full(N, T, dtype=int)
    for i in range(N):
        k = T
        for j in range(T - 1, -1, -1):
            if preds[i, j] == final[i]:
                k = j + 1
            else:
                break
        teq[i] = k

    stable = (teq <= 2) & (final == gt)
    first_corr = np.full(N, T + 1)
    for i in range(N):
        idx = np.where(corr[i])[0]
        if len(idx):
            first_corr[i] = idx[0] + 1
    rescue = (final == gt) & (first_corr >= 3)
    harmful = (corr[:, :-1].any(axis=1)) & (final != gt)
    oscill = flips >= 3

    picks = {}
    for name, mask in [("stable", stable), ("rescue", rescue),
                       ("harmful", harmful), ("oscillatory", oscill)]:
        idx = np.where(mask)[0]
        picks[name] = idx[:args.n_per_group].tolist()
        print(f"{name:12s} n={mask.sum():4d} picked={picks[name]}")

    with open(os.path.join(base, f"cases_{args.proto}.json"), "w") as f:
        json.dump({k: [int(i) for i in v] for k, v in picks.items()}, f, indent=1)

    # ---- 轨迹图 ----
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (name, idxs) in zip(axes, picks.items()):
        for rank, i in enumerate(idxs):
            y = preds[i] - gt[i]  # 0 = 正确（GT 类），其他 = 错类偏移
            label = "correct (0)" if final[i] == gt[i] else "final wrong"
            ax.plot(np.arange(1, T + 1), y, "o-", lw=1, ms=3, alpha=0.85,
                    label=f"#{i} ({'acc' if final[i]==gt[i] else 'err'})")
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_title(f"{name} (n={len(idxs)})")
        ax.set_xlabel("scale t")
        ax.set_xticks(range(1, T + 1))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("prediction offset vs GT (0 = correct)")
    fig.suptitle(f"{args.dataset} / {args.backbone} / proto={args.proto} — prediction trajectories",
                 y=1.04)
    out_dir = os.path.join(PROJECT_ROOT, "figures", "cases")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.dataset}_{tag}_{args.proto}_cases"
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
