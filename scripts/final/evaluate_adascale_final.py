#!/usr/bin/env python3
"""AdaScale v2 正式评估：数据集级模式选择 + 样本级早停。

机制（全部 training-free / label-free）：
  1. 校准（n 个未标注样本）：
       若 mean H(fused-10) > mean H(single-10)  -> SINGLE 模式
       否则                                      -> FUSED 模式
  2. 执行：
       SINGLE: 仅计算最大 scale（1 次前向），输出其预测
       FUSED : AdaScale 稳定性早停（冻结配置 r=4, delta=0.65, eps=0.01）

输出：outputs/final/adascale_v2_<ds>_<bk>_<proto>.json
     + 每样本预测 parquet（用于进一步统计）

用法:
    python scripts/final/evaluate_adascale_final.py --dataset PET --proto text
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
from src.adascale.simulate import StabilityPolicy, FixedKPolicy, simulate  # noqa: E402

LOGIT_TAU = 0.01
CAL_N = 500
CAL_SEED = 0


def load_all(dataset, backbone, proto):
    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, dataset, backbone)
    if proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(root, dataset, backbone, "", 10)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    return feats, labels.numpy(), prototypes


def fused_single_predictions(feats, prototypes):
    """full-10 融合预测 与 single-10 预测 + 各自熵。"""
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
    Hf = -(pf * torch.log(pf.clamp_min(1e-12))).sum(-1).numpy()
    Hs = -(ps * torch.log(ps.clamp_min(1e-12))).sum(-1).numpy()
    return (lf.argmax(-1).numpy(), ls.argmax(-1).numpy(), Hf, Hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--n", type=int, default=CAL_N)
    ap.add_argument("--seed", type=int, default=CAL_SEED)
    ap.add_argument("--theta", type=float, default=0.01,
                    help="relative entropy-difference threshold for SINGLE mode")
    ap.add_argument("--stab", default="4,0.65,0.01",
                    help="stability policy r,delta,eps for FUSED mode")
    args = ap.parse_args()
    r_s, d_s, e_s = [float(x) for x in args.stab.split(",")]
    stab_r, stab_delta, stab_eps = int(r_s), d_s, e_s

    feats, labels, prototypes = load_all(args.dataset, args.backbone, args.proto)
    N = len(labels)
    print(f"[adascale-v2] {args.dataset}/{args.backbone}/{args.proto} N={N}")

    pred_f, pred_s, Hf, Hs = fused_single_predictions(feats, prototypes)

    # ---- 校准（模式选择；label-free）----
    # 判据：相对熵差 (mean H_fused - mean H_single) / log C > theta
    #   theta=0.01（先验：仅在融合把不确定性抬高 ≥1% 最大熵时才干预）
    theta = args.theta
    C = prototypes.shape[0]
    logC = float(np.log(C))
    rng = np.random.default_rng(args.seed)
    cal_idx = rng.choice(N, size=min(args.n, N), replace=False)
    rel_entropy_diff = float((Hf[cal_idx] - Hs[cal_idx]).mean() / logC)
    mode = "SINGLE" if rel_entropy_diff > theta else "FUSED"
    print(f"  calibration n={len(cal_idx)}: rel._H_diff/logC={rel_entropy_diff:+.5f} "
          f"(theta={theta}) -> mode={mode}")

    # ---- 执行 ----
    if mode == "SINGLE":
        pred_v2 = pred_s.copy()
        scales_v2 = np.ones(N, dtype=np.int64)  # 仅 1 次前向（最大 scale）
        exit_hist = {1: N}
    else:
        policy = StabilityPolicy(r=stab_r, delta=stab_delta, eps=stab_eps,
                                 min_scale=1, use_js=True, use_margin=True)
        res = simulate(feats, prototypes, policy)
        pred_v2 = res.pred
        scales_v2 = res.scales_used
        exit_hist = res.exit_hist

    acc_v2 = float((pred_v2 == labels).mean())
    acc_f = float((pred_f == labels).mean())
    acc_s = float((pred_s == labels).mean())
    acc_v2_excl = float((pred_v2[np.setdiff1d(np.arange(N), cal_idx)] ==
                         labels[np.setdiff1d(np.arange(N), cal_idx)]).mean())

    # ---- 对照：Fixed-K ----
    fixed = {}
    for k in (5, 7, 9, 10):
        r = simulate(feats, prototypes, FixedKPolicy(k))
        fixed[f"fixed-{k}"] = {"acc": float((r.pred == labels).mean()),
                               "avg_scales": float(k)}

    out = {
        "dataset": args.dataset, "backbone": args.backbone, "proto": args.proto,
        "N": N, "cal_n": len(cal_idx), "mode": mode, "theta": args.theta,
        "stab": args.stab,
        "rel_entropy_diff_cal": rel_entropy_diff,
        "acc_full10_fused": acc_f, "acc_single10": acc_s,
        "adascale_v2": {
            "acc": acc_v2,
            "acc_excl_cal": acc_v2_excl,
            "avg_scales": float(scales_v2.mean()),
            "exit_hist": {str(k): int(v) for k, v in sorted(exit_hist.items())},
        },
        "fixed": fixed,
    }
    out_dir = ROOT / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stab_tag = f"r{stab_r}_d{stab_delta}_e{stab_eps}"
    f_tag = f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_{stab_tag}"
    if mode == "SINGLE":
        f_tag = f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_single"
    with open(out_dir / f"adascale_v2_{f_tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    np.savez(out_dir / f"adascale_v2_{f_tag}_preds.npz",
             labels=labels, pred_v2=pred_v2, scales_v2=scales_v2,
             pred_fused=pred_f, pred_single=pred_s)

    print(f"  full-10 fused: {acc_f*100:.2f}% | single-10: {acc_s*100:.2f}%")
    print(f"  AdaScale v2:   {acc_v2*100:.2f}% @ {scales_v2.mean():.2f} scales "
          f"(excl cal: {acc_v2_excl*100:.2f}%)")
    print(f"  saved {out_dir}/adascale_v2_{f_tag}.json")


if __name__ == "__main__":
    main()
