#!/usr/bin/env python3
"""融合-单尺度仲裁机制探索（研究分析，非正式方法）。

探索候选判据（统一规则，无 per-dataset 调参）:
  R1: margin 选择       argsort margin: fused vs single_t
  R2: agreement-gated   若 fused 与 single_t 一致 -> fused；否则 margin 高者
  R3: logit 选择        融合 top1 logit vs 单 top1 logit
  R4: stable-single     若 fused != single_t 且 single margin > 0.5 -> single

输出各 t 的 acc（供机制设计参考）。

用法:
    python scripts/final/arbitration_probe.py --dataset EUROSAT --proto text
"""
from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    image_root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(image_root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(image_root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    labels = labels.numpy()

    T = len(feats)
    prefix_sum = None
    print(f"[arbitration] {args.dataset}/{args.backbone}/{args.proto}")
    print(f"{'t':>2} {'fused':>7} {'single':>7} {'R1':>7} {'R2':>7} {'R3':>7} "
          f"{'R4':>7} {'R5':>7} {'R6':>7}")
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_sum = SCALE_WEIGHTS[t - 1] * f if prefix_sum is None else \
            prefix_sum + SCALE_WEIGHTS[t - 1] * f
        fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        logits_f = fn @ prototypes.T
        logits_s = f @ prototypes.T
        prob_f = F.softmax(logits_f / LOGIT_TAU, dim=-1)
        prob_s = F.softmax(logits_s / LOGIT_TAU, dim=-1)
        pred_f = logits_f.argmax(-1)
        pred_s = logits_s.argmax(-1)

        top2f = torch.topk(prob_f, 2, dim=-1).values
        top2s = torch.topk(prob_s, 2, dim=-1).values
        mf = top2f[:, 0] - top2f[:, 1]
        ms = top2s[:, 0] - top2s[:, 1]

        acc_f = (pred_f.numpy() == labels).mean()
        acc_s = (pred_s.numpy() == labels).mean()

        # R1: margin 大者
        pick_s = ms > mf
        pred_r1 = torch.where(pick_s, pred_s, pred_f)
        acc_r1 = (pred_r1.numpy() == labels).mean()

        # R2: 一致 -> fused；否则 margin 大者
        agree = pred_f == pred_s
        pred_r2 = torch.where(agree, pred_f, torch.where(pick_s, pred_s, pred_f))
        acc_r2 = (pred_r2.numpy() == labels).mean()

        # R3: top1 logit 大者
        l1f = logits_f.max(-1).values
        l1s = logits_s.max(-1).values
        pick_s3 = l1s > l1f
        pred_r3 = torch.where(pick_s3, pred_s, pred_f)
        acc_r3 = (pred_r3.numpy() == labels).mean()

        # R4: 不一致时若 single margin>0.5 用 single
        pred_r4 = torch.where(~agree & (ms > 0.5), pred_s, pred_f)
        acc_r4 = (pred_r4.numpy() == labels).mean()

        # R5: logit 空间软加权 (beta = m_s/(m_f+m_s))
        beta = (ms / (mf + ms + 1e-9)).unsqueeze(-1)
        logits_r5 = (1 - beta) * logits_f + beta * logits_s
        acc_r5 = (logits_r5.argmax(-1).numpy() == labels).mean()

        # R6: prob 空间软加权
        prob_r6 = (1 - beta) * prob_f + beta * prob_s
        acc_r6 = (prob_r6.argmax(-1).numpy() == labels).mean()

        print(f"{t:>2} {acc_f*100:>6.2f}% {acc_s*100:>6.2f}% "
              f"{acc_r1*100:>6.2f}% {acc_r2*100:>6.2f}% {acc_r3*100:>6.2f}% "
              f"{acc_r4*100:>6.2f}% {acc_r5*100:>6.2f}% {acc_r6*100:>6.2f}%")


if __name__ == "__main__":
    main()
