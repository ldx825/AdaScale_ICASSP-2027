#!/usr/bin/env python3
"""模式判据校准稳健性验证。

场景：部署时可用 n 个未标注校准样本估计判据；实际评估在全测试集。
本脚本模拟：
  1. 全量测试集上计算实际最优模式（by acc）
  2. 随机抽 n 个校准样本，计算判据统计量并做出模式预测
  3. 重复 reps 次，统计判据正确率

判据候选：
  C1 (sign test):  frac(m_single > m_fused) > 0.5 -> SINGLE
  C2 (entropy):    mean(H_fused - H_single) > 0   -> SINGLE
  C3 (margin):     mean(m_single - m_fused) > 0   -> SINGLE

用法:
    python scripts/final/mode_calibration_sim.py --dataset PET --proto text \
        --n 500 --reps 200
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


def per_sample_quantities(feats, prototypes):
    """返回每样本: mf, ms, Hf, Hs, pred_f, pred_s。"""
    T = len(feats)
    prefix_sum = None
    for i in range(T):
        f = feats[i]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_sum = SCALE_WEIGHTS[i] * f if prefix_sum is None else \
            prefix_sum + SCALE_WEIGHTS[i] * f
    fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fs = feats[-1] / feats[-1].norm(dim=-1, keepdim=True).clamp_min(1e-12)

    lf = fn @ prototypes.T
    ls = fs @ prototypes.T
    pf = F.softmax(lf / LOGIT_TAU, dim=-1)
    ps = F.softmax(ls / LOGIT_TAU, dim=-1)
    t2f = torch.topk(pf, 2, dim=-1).values
    t2s = torch.topk(ps, 2, dim=-1).values
    mf = (t2f[:, 0] - t2f[:, 1]).numpy()
    ms = (t2s[:, 0] - t2s[:, 1]).numpy()
    Hf = -(pf * torch.log(pf.clamp_min(1e-12))).sum(-1).numpy()
    Hs = -(ps * torch.log(ps.clamp_min(1e-12))).sum(-1).numpy()
    pred_f = lf.argmax(-1).numpy()
    pred_s = ls.argmax(-1).numpy()
    return mf, ms, Hf, Hs, pred_f, pred_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    labels = labels.numpy()

    mf, ms, Hf, Hs, pred_f, pred_s = per_sample_quantities(feats, prototypes)
    N = len(labels)
    actual_better = "SINGLE" if (pred_s == labels).mean() > (pred_f == labels).mean() \
        else "FUSED"

    rng = np.random.default_rng(args.seed)
    ok = {"C1_sign": 0, "C2_entropy": 0, "C3_margin": 0}
    stats = {"C1_sign": [], "C2_entropy": [], "C3_margin": []}
    for _ in range(args.reps):
        idx = rng.choice(N, size=args.n, replace=False)
        c1 = (ms[idx] > mf[idx]).mean()
        c2 = (Hf[idx] - Hs[idx]).mean()
        c3 = (ms[idx] - mf[idx]).mean()
        pred = lambda v: "SINGLE" if v > 0.5 else "FUSED"  # noqa: E731
        # C1: >0.5 -> SINGLE
        if pred(c1) == actual_better:
            ok["C1_sign"] += 1
        # C2: >0 -> SINGLE (continuous)
        if ("SINGLE" if c2 > 0 else "FUSED") == actual_better:
            ok["C2_entropy"] += 1
        # C3: >0 -> SINGLE
        if ("SINGLE" if c3 > 0 else "FUSED") == actual_better:
            ok["C3_margin"] += 1
        stats["C1_sign"].append(c1)
        stats["C2_entropy"].append(c2)
        stats["C3_margin"].append(c3)

    print(f"[mode-calib] {args.dataset}/{args.proto} n={args.n} reps={args.reps} "
          f"actual={actual_better}")
    for k in ok:
        vals = np.array(stats[k])
        print(f"  {k:12s} accuracy={ok[k]/args.reps*100:5.1f}%  "
              f"stat mean={vals.mean():+.4f} sd={vals.std():.4f}")

    out = ROOT / "outputs" / "mode_selector" / \
        f"calib_{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_n{args.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n": args.n, "reps": args.reps, "actual": actual_better,
                   "accuracy": {k: ok[k] / args.reps for k in ok},
                   "stat_mean": {k: float(np.mean(stats[k])) for k in stats},
                   "stat_sd": {k: float(np.std(stats[k])) for k in stats}}, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
