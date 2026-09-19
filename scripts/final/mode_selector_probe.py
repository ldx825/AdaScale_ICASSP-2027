#!/usr/bin/env python3
"""模式选择判据探针：寻找稳健的"融合 vs 单尺度"无标签判据。

在 (dataset, proto) 设置上计算多种统计量，检验它们能否正确指示
"fused 更好"还是 "single(s10) 更好"：

  S1: mean(m_s - m_f)              margin 均值差
  S2: frac(m_s > m_f)              样本比例
  S3: mean(gap_s - gap_f)          logit 空间 gap (z1 - z2) 差
  S4: mean(H_f - H_s)              熵差（正=单尺度更确定）
  S5: frac(argmax_f == argmax_s)   预测一致率
  S6: mean(m_s - m_f) / mean(m_f)  相对 margin 差

输出: outputs/mode_selector/<ds>_<bk>_<proto>.json + 汇总打印

用法:
    python scripts/final/mode_selector_probe.py --dataset PET --proto text
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


def compute_stats(feats, prototypes, labels):
    T = len(feats)
    prefix_sum = None
    for i in range(T):
        f = feats[i]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_sum = SCALE_WEIGHTS[i] * f if prefix_sum is None else \
            prefix_sum + SCALE_WEIGHTS[i] * f
    fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    lf = fn @ prototypes.T                      # fused logits
    ls = (feats[-1] / feats[-1].norm(dim=-1, keepdim=True).clamp_min(1e-12)) @ prototypes.T

    pf = F.softmax(lf / LOGIT_TAU, dim=-1)
    ps = F.softmax(ls / LOGIT_TAU, dim=-1)
    t2f = torch.topk(pf, 2, dim=-1).values
    t2s = torch.topk(ps, 2, dim=-1).values
    mf = (t2f[:, 0] - t2f[:, 1]).numpy()
    ms = (t2s[:, 0] - t2s[:, 1]).numpy()

    # logit gaps
    l2f = torch.topk(lf, 2, dim=-1).values
    l2s = torch.topk(ls, 2, dim=-1).values
    gapf = (l2f[:, 0] - l2f[:, 1]).numpy()
    gaps = (l2s[:, 0] - l2s[:, 1]).numpy()

    # entropy
    Hf = -(pf * torch.log(pf.clamp_min(1e-12))).sum(-1).numpy()
    Hs = -(ps * torch.log(ps.clamp_min(1e-12))).sum(-1).numpy()

    amf = lf.argmax(-1).numpy()
    ams = ls.argmax(-1).numpy()

    acc_f = float((amf == labels).mean())
    acc_s = float((ams == labels).mean())

    stats = {
        "acc_fused": acc_f,
        "acc_single": acc_s,
        "S1_mean_margin_diff": float((ms - mf).mean()),
        "S2_frac_ms_gt_mf": float((ms > mf).mean()),
        "S3_mean_logitgap_diff": float((gaps - gapf).mean()),
        "S4_mean_entropy_diff": float((Hf - Hs).mean()),
        "S5_pred_agreement": float((amf == ams).mean()),
        "S6_rel_margin_diff": float((ms - mf).mean() / (mf.mean() + 1e-12)),
        "mean_margin_fused": float(mf.mean()),
        "mean_margin_single": float(ms.mean()),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)

    stats = compute_stats(feats, prototypes, labels.numpy())

    out = ROOT / "outputs" / "mode_selector" / \
        f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)

    better = "FUSED" if stats["acc_fused"] >= stats["acc_single"] else "SINGLE"
    print(f"[mode-probe] {args.dataset}/{args.proto}: actually better={better} "
          f"(fused={stats['acc_fused']*100:.2f}% single={stats['acc_single']*100:.2f}%)")
    for k, v in stats.items():
        if k.startswith(("S", "mean_", "acc_")):
            print(f"    {k:26s} {v:+.5f}" if isinstance(v, float) else f"    {k:26s} {v}")


if __name__ == "__main__":
    main()
