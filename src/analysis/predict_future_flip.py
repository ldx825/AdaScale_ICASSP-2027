"""
predict_future_flip.py
----------------------
分析：在 scale t 处，哪些 label-free 信号可以预测「继续计算是否会改变预测」。

对每个样本 i、每个 t ∈ [2, 9]:
  target(i,t) = 1 if 存在 j > t 使得 y_j != y_t   （即"若在 t 退出则与 full-10 预测不同"）
  signals(i,t) = margin_t, entropy_t, js_t (相对 t-1), 连续稳定长度, raw margin_t ...

输出:
  - 每个 t 各信号的 ROC-AUC（预测 future flip）
  - 组合规则示例（margin 阈值 / stability 阈值）的覆盖率-精度曲线（用于方法设计）

用法:
  python src/analysis/predict_future_flip.py --dataset PET --backbone ViT-B/32 --proto text
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def roc_auc(score, label):
    """无依赖 AUC（label: 1=positive）。score 越大越倾向 positive。"""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # 处理 ties（用平均秩近似即可，这里简单处理）
    pos = label == 1
    n_pos = pos.sum()
    n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    tag = args.backbone.replace("/", "")
    base = os.path.join(PROJECT_ROOT, "outputs", "scale_audit", args.dataset, tag)
    df = pd.read_parquet(os.path.join(base, f"per_sample_{args.proto}.parquet"))

    N = df.sample_id.nunique()
    T = df.scale_id.max()
    pv = df.pivot(index="sample_id", columns="scale_id", values="top1_class")
    preds = pv[np.arange(1, T + 1)].values  # (N, T)

    results = {}
    print(f"== future-flip predictability ({args.dataset}/{args.backbone}/{args.proto}) ==")
    print("（AUC > 0.5 = 信号越大越稳定；stable_len/margin 等均为正向稳定性指标）")
    for t in range(2, T):  # 在 t 退出，预测 t 后是否有翻转
        fut = (preds[:, t:] != preds[:, [t - 1]]).any(axis=1)  # j>t 中是否有翻转
        stable = (~fut).astype(int)                            # 1 = 退出安全（不再变化）
        # 注意: preds 列索引 t-1 = scale t; j>t 对应列索引 >= t
        sub = df[df.scale_id == t].set_index("sample_id").loc[range(N)]
        signals = {
            "margin": sub["margin"].values,
            "raw_margin": sub["raw_margin"].values,
            "neg_entropy": -sub["entropy"].values,
            "neg_js": -sub["js_divergence_to_prev"].values,
        }
        # 连续稳定长度: t 及之前 → 预测与 t 相同的最长后缀（含 t）
        stable_len = np.ones(N)
        for i in range(N):
            k = 1
            while t - 1 - k >= 0 and preds[i, t - 1 - k] == preds[i, t - 1]:
                k += 1
            stable_len[i] = k
        signals["stable_len"] = stable_len
        aucs = {k: roc_auc(v, stable) for k, v in signals.items()}
        pos_rate = fut.mean()
        results[t] = {"auc_stability": aucs, "future_flip_rate": float(pos_rate)}
        print(f"  t={t}: future-flip rate={pos_rate*100:.1f}% | " +
              " | ".join(f"{k}: {v:.3f}" for k, v in aucs.items()))

    out = os.path.join(base, f"predict_future_flip_{args.proto}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
