#!/usr/bin/env python3
"""Per-scale 融合 vs 单尺度预测对比分析。

对每个 t 计算：acc(prefix_t 融合) vs acc(single_t)，以及"若退出在 t 时输出
单尺度 vs 输出融合"的差异。揭示数据集依赖的 scale utility 结构。

用法:
    python scripts/final/fusion_vs_single.py --dataset EUROSAT --proto text
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate, SCALE_WEIGHTS,
)

LOGIT_TAU = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--ngen", type=int, default=10)
    args = ap.parse_args()

    image_root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(image_root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(image_root, args.dataset, args.backbone, "", args.ngen)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    labels = labels.numpy()

    T = len(feats)
    row = []
    prefix_sum = None
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_sum = SCALE_WEIGHTS[t - 1] * f if prefix_sum is None else \
            prefix_sum + SCALE_WEIGHTS[t - 1] * f
        fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits_fused = (fn @ prototypes.T)
        logits_single = (f @ prototypes.T)
        pred_fused = logits_fused.argmax(-1).numpy()
        pred_single = logits_single.argmax(-1).numpy()
        prob_f = F.softmax(logits_fused / LOGIT_TAU, dim=-1)
        top2 = torch.topk(prob_f, 2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).numpy()
        acc_f = (pred_fused == labels).mean()
        acc_s = (pred_single == labels).mean()
        agree = (pred_fused == pred_single).mean()
        # 逐样本 margin 高的 scale 选谁（融合 margin vs 单 margin）
        row.append(dict(
            t=t,
            acc_fused=float(acc_f),
            acc_single=float(acc_s),
            diff=float(acc_f - acc_s),
            agree=float(agree),
            margin_med=float(np.median(margin)),
        ))

    print(f"[fusion_vs_single] {args.dataset}/{args.backbone}/{args.proto}")
    print(f"{'t':>2} {'acc_fused':>10} {'acc_single':>10} {'diff':>8} {'agree':>7} {'margin_med':>10}")
    for r in row:
        print(f"{r['t']:>2} {r['acc_fused']*100:>9.2f}% {r['acc_single']*100:>9.2f}% "
              f"{r['diff']*100:>+7.2f}% {r['agree']*100:>6.1f}% {r['margin_med']:>10.3f}")

    out = ROOT / "outputs" / "fusion_vs_single" / \
        f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
