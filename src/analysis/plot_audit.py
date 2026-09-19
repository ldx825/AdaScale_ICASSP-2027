"""
plot_audit.py
-------------
Scale Utility Audit 图表（Fig A–F，文档 §11）。

输入: outputs/scale_audit/<dataset>/<backbone>/{per_sample,summary}_<proto>.{parquet,json}
输出: figures/audit/<dataset>_<backbone>_<proto>_fig{A..F}.{pdf,png}

用法:
  python src/analysis/plot_audit.py --dataset PET --backbone ViT-B/32 --proto text
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "audit")

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(dataset, backbone, proto, root=None):
    tag = backbone.replace("/", "")
    base = os.path.join(root or os.path.join(PROJECT_ROOT, "outputs", "scale_audit"),
                        dataset, tag)
    df = pd.read_parquet(os.path.join(base, f"per_sample_{proto}.parquet"))
    with open(os.path.join(base, f"summary_{proto}.json")) as f:
        sm = json.load(f)
    return df, sm


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print("  saved", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text")
    args = ap.parse_args()

    df, sm = load(args.dataset, args.backbone, args.proto)
    N = sm["num_samples"]
    T = sm["num_scales"]
    tag = f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}"
    scales = np.arange(1, T + 1)

    acc = np.array(sm["prefix_acc_curve"]) * 100
    acc_ind = np.array(sm["per_scale_independent_acc"]) * 100

    # ---------- Fig A: Accuracy vs Number of Scales ----------
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(scales, acc, "o-", label="Prefix (cumulative)")
    ax.plot(scales, acc_ind, "s--", alpha=0.6, label="Single scale")
    ax.set_xlabel("# scales evaluated")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{args.dataset} / {args.backbone} / {args.proto}")
    ax.legend()
    ax.set_xticks(scales)
    save(fig, f"{tag}_figA_acc_vs_scales")

    # ---------- Fig B: CDF of t_eq ----------
    teq = df[df.scale_id == T].reset_index(drop=True)  # 无直接 t_eq 列，重算:
    # t_eq 分布从 summary 读取更直接
    hist = sm["teq_hist"]
    counts = np.array([hist.get(str(k), 0) for k in scales], dtype=float)
    cdf = np.cumsum(counts) / counts.sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(scales, cdf * 100, "o-")
    for k in range(1, T + 1):
        if cdf[k - 1] > 0.5:
            ax.axvline(k, color="gray", ls=":", lw=0.8)
            ax.annotate(f"median={k}", (k, cdf[k - 1]), fontsize=8,
                        xytext=(k + 0.3, cdf[k - 1] - 8))
            break
    ax.set_xlabel("t_eq (min final-equivalent scale)")
    ax.set_ylabel("CDF (% of samples)")
    ax.set_title("Stabilization distribution")
    ax.set_xticks(scales)
    save(fig, f"{tag}_figB_teq_cdf")

    # ---------- Fig C: Flip rate vs transition ----------
    fr = np.array(sm["flip_rate_per_transition"]) * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(np.arange(2, T + 1), fr)
    ax.set_xlabel("transition t-1 → t")
    ax.set_ylabel("flip rate (%)")
    ax.set_title("Prediction flip rate per transition")
    ax.set_xticks(np.arange(2, T + 1))
    save(fig, f"{tag}_figC_flip_rate")

    # ---------- Fig D: harmful vs rescue transitions ----------
    ht = np.array(sm["harmful_transitions"], dtype=float) / N * 100
    rt = np.array(sm["rescue_transitions"], dtype=float) / N * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    w = 0.4
    ax.bar(np.arange(2, T + 1) - w / 2, ht, width=w, label="correct→wrong (harmful)")
    ax.bar(np.arange(2, T + 1) + w / 2, rt, width=w, label="wrong→correct (rescue)")
    ax.set_xlabel("transition t-1 → t")
    ax.set_ylabel("% of samples")
    ax.set_title("Harmful vs rescue transitions")
    ax.set_xticks(np.arange(2, T + 1))
    ax.legend(fontsize=8)
    save(fig, f"{tag}_figD_harmful_rescue")

    # ---------- Fig E: margin / entropy by group ----------
    # t_eq per sample
    hist = sm["teq_hist"]
    # 重新计算每样本 t_eq（从 per_sample 推断）
    pv = df.pivot(index="sample_id", columns="scale_id", values="top1_class")
    final = pv[T].values
    preds = pv[scales].values
    teq_per = np.full(N, T, dtype=int)
    for i in range(N):
        t = T
        for j in range(T - 1, -1, -1):
            if preds[i, j] == final[i]:
                t = j + 1
            else:
                break
        teq_per[i] = t
    corr = df[df.scale_id == T].sort_values("sample_id")["correct"].values
    gt = df[df.scale_id == T].sort_values("sample_id")["ground_truth"].values
    # harmful: 早期某 scale 正确，最终错误; rescue: 最终正确且早期错误
    corr_all = df.pivot(index="sample_id", columns="scale_id", values="correct")
    corr_all = corr_all[scales].values.astype(bool)
    harmful = corr_all[:, :-1].any(axis=1) & (~corr_all[:, -1])
    rescue = corr_all[:, -1] & (~corr_all[:, :-1].all(axis=1))
    stable = teq_per <= 3

    m1 = df[(df.scale_id == 2)].sort_values("sample_id")["margin"].values
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    groups = [("stable (t_eq≤3)", stable), ("rescue", rescue), ("harmful", harmful)]
    data = [m1[g] for _, g in groups if g.sum() > 0]
    labels = [f"{n}\n(n={g.sum()})" for n, g in groups if g.sum() > 0]
    if data:
        axes[0].boxplot(data, tick_labels=labels, showfliers=False)
        axes[0].set_ylabel("margin at scale 2")
    e1 = df[(df.scale_id == 2)].sort_values("sample_id")["entropy"].values
    data_e = [e1[g] for _, g in groups if g.sum() > 0]
    if data_e:
        axes[1].boxplot(data_e, tick_labels=labels, showfliers=False)
        axes[1].set_ylabel("entropy at scale 2")
    fig.suptitle(f"Early (scale-2) signals by outcome group — {args.dataset}", y=1.02)
    save(fig, f"{tag}_figE_signal_by_group")

    # ---------- Fig F: JS divergence / feature cosine vs scale ----------
    js = df.groupby("scale_id")["js_divergence_to_prev"].mean()
    fc = df.groupby("scale_id")["feature_cosine_to_prev"].mean()
    fs1 = df.groupby("scale_id")["feature_cosine_to_scale1"].mean()
    x = np.arange(2, T + 1)  # scale 2..T（JS/cos 相对上一 scale 的首个有效点为 scale 2）
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(x, js.loc[2:].values, "o-", label="JS(p_t, p_{t-1})")
    ax2 = ax.twinx()
    ax2.plot(x, fc.loc[2:].values, "s--", color="tab:green",
             label="cos(f_t, f_{t-1})")
    ax2.plot(x, fs1.loc[2:].values, "^:", color="tab:red",
             label="cos(f_t, f_1)")
    ax.set_xlabel("scale t")
    ax.set_ylabel("JS divergence (left)")
    ax2.set_ylabel("feature cosine (right)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.set_title("Distribution / feature drift vs scale")
    ax.set_xticks(scales)
    save(fig, f"{tag}_figF_drift_vs_scale")

    print("done.")


if __name__ == "__main__":
    main()
