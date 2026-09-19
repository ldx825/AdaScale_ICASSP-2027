#!/usr/bin/env python3
"""深层逐样本诊断：四象限正确性 × 融合行为、margin/熵轨迹、特征几何。

三组分析（全部缓存特征，秒级）：
  Q. 四象限: (s1 正确?, s10 正确?) 交叉表 + 每象限 f 融合/单尺度行为
  T. 轨迹: prefix margin / 熵 的 mean±sem 曲线（早退 vs 走满；走满内 correct vs wrong）
  G. 几何: cos(f_s, p_gt) 与 cos(f_s, p_gt) - cos(f_s, p_top1err) 随尺度

输出: outputs/deep_diag/<DS>_<proto>.json + figures/deep_diag/*.png/pdf

用法: python scripts/analysis/deep_diag.py --dataset EUROSAT --proto text
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


def load_all(dataset, backbone, proto):
    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, dataset, backbone)
    if proto == "text":
        prototypes = text_emb
    else:
        gf, _ = load_gen_scales(root, dataset, backbone, "", 10)
        prototypes = full_scale_aggregate(gf).mean(dim=1)
    return feats, labels.numpy(), prototypes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    args = ap.parse_args()

    feats, labels, protos = load_all(args.dataset, args.backbone, args.proto)
    N = len(labels)
    T = len(feats)
    BK = args.backbone.replace("/", "")
    print(f"[deep-diag] {args.dataset}/{BK}/{args.proto} N={N}")

    F_all = torch.stack([f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                         for f in feats])                     # (T,N,D)
    logits_all = torch.einsum("tnd,cd->tnc", F_all, protos)   # 原始内积
    P_all = F.softmax(logits_all / LOGIT_TAU, dim=-1)         # (T,N,C)
    pred_all = logits_all.argmax(-1)                          # (T,N)
    H_all = -(P_all * torch.log(P_all.clamp_min(1e-12))).sum(-1)  # (T,N)
    top2 = torch.topk(P_all, 2, dim=-1).values
    M_all = top2[..., 0] - top2[..., 1]

    corr_all = (pred_all.numpy() == labels[None, :])          # (T,N) 单视图
    c1, c10 = corr_all[0], corr_all[-1]

    # ---- 前缀融合轨迹 ----
    pw = SCALE_WEIGHTS.astype(np.float32)
    prefix_sum = torch.zeros(N, F_all.shape[2])
    prefix_pred = np.zeros((T, N), dtype=np.int64)
    prefix_M = np.zeros((T, N), dtype=np.float32)
    prefix_H = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        prefix_sum += pw[t] * F_all[t]
        fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lg = fn @ protos.T
        P = F.softmax(lg / LOGIT_TAU, dim=-1)
        prefix_pred[t] = lg.argmax(-1).numpy()
        t2 = torch.topk(P, 2, dim=-1).values
        prefix_M[t] = (t2[:, 0] - t2[:, 1]).numpy()
        prefix_H[t] = -(P * torch.log(P.clamp_min(1e-12))).sum(-1).numpy()
    fused_corr = (prefix_pred[-1] == labels)
    fused_acc = float(fused_corr.mean())

    # ---- Q. 四象限 ----
    quad_names = ["QQ", "QN", "NQ", "NN"]  # (s1, s10)
    quad = {}
    for name, m1, m10 in [("QQ", True, True), ("QN", True, False),
                          ("NQ", False, True), ("NN", False, False)]:
        sel = np.where(c1 == m1)[0]
        sel = sel[np.where(c10[sel] == m10)[0]] if len(sel) else sel
        if len(sel) == 0:
            quad[name] = {"n": 0}
            continue
        quad[name] = {
            "n": int(len(sel)),
            "frac": float(len(sel) / N),
            "fused_acc": float(fused_corr[sel].mean()),
            "s1_acc": float(c1[sel].mean()),
            "s10_acc": float(c10[sel].mean()),
        }
    print("  quadrants:", {k: v.get("n", 0) for k, v in quad.items()})
    for k, v in quad.items():
        if v["n"]:
            print(f"    {k}: n={v['n']:5d} frac={v['frac']*100:5.1f}% "
                  f"fused={v['fused_acc']*100:5.1f} s1={v['s1_acc']*100:5.1f} "
                  f"s10={v['s10_acc']*100:5.1f}")

    # ---- D. label-free 拖拽轨迹（在 s1 vs s10 预测不一致样本上）----
    p1_np = pred_all.numpy()[0]
    p10_np = pred_all.numpy()[-1]
    disagree = p1_np != p10_np
    n_dis = int(disagree.sum())
    agree10 = np.zeros((T, n_dis), dtype=np.float32)   # prefix fused == s10
    agree1 = np.zeros((T, n_dis), dtype=np.float32)    # prefix fused == s1
    for t in range(T):
        agree10[t] = (prefix_pred[t][disagree] == p10_np[disagree])
        agree1[t] = (prefix_pred[t][disagree] == p1_np[disagree])
    drag = {
        "n_disagree": n_dis,
        "to_s10_curve": agree10.mean(1).tolist(),
        "to_s1_curve": agree1.mean(1).tolist(),
        "final_to_s10": float(agree10[-1].mean()),
        "final_to_s1": float(agree1[-1].mean()),
    }
    print(f"  drag: n_dis={n_dis} final to_s10={drag['final_to_s10']*100:.1f}% "
          f"to_s1={drag['final_to_s1']*100:.1f}%")

    # ---- T. 轨迹分组 ----
    npz_p = ROOT / "outputs" / "final" / \
        f"adascale_v2_{args.dataset}_{BK}_{args.proto}_r4_d0.65_e0.01_preds.npz"
    if not npz_p.exists():
        npz_p = ROOT / "outputs" / "final" / \
            f"adascale_v2_{args.dataset}_{BK}_{args.proto}_single_preds.npz"
    early = full = None
    if npz_p.exists():
        z = np.load(npz_p)
        sc = z["scales_v2"]
        early = sc < T
        full = ~early
        print(f"  early={early.sum()} full={full.sum()}")

    # ---- G. 几何 ----
    cos_gt = torch.einsum("tnd,nd->tn", F_all, protos[labels])          # (T,N)
    top1_pred_all = pred_all.numpy()
    # cos 到"该样本在 s10 的 top1 预测类的原型"
    p10 = torch.from_numpy(top1_pred_all[-1])
    cos_p10 = torch.einsum("tnd,nd->tn", F_all, protos[p10])
    # 几何 margin: cos_gt - max_{c!=gt} cos
    cos_all = torch.einsum("tnd,cd->tnc", F_all, protos)
    cos_gt_full = cos_all.gather(
        2, torch.from_numpy(labels)[None, :, None].repeat(T, 1, 1)).squeeze(-1)
    cos_all_masked = cos_all.clone()
    cos_all_masked.scatter_(2, torch.from_numpy(labels)[None, :, None].repeat(T, 1, 1), -1e9)
    cos_best_other = cos_all_masked.max(-1).values
    geo_margin = cos_gt_full - cos_best_other                       # (T,N)

    out = {
        "dataset": args.dataset, "backbone": args.backbone, "proto": args.proto,
        "N": int(N), "fused_acc": fused_acc,
        "quadrants": quad,
        "drag": drag,
        "trajectory": {
            "prefix_M_mean": prefix_M.mean(1).tolist(),
            "prefix_H_mean": prefix_H.mean(1).tolist(),
        },
        "geometry": {
            "cos_gt_mean_by_scale": cos_gt_full.mean(1).tolist(),
            "geo_margin_mean_by_scale": geo_margin.mean(1).tolist(),
        },
    }
    # 轨迹分组统计（供画图）
    traj = {}
    if early is not None:
        for name, sel in [("early", early), ("full_correct", full & fused_corr),
                          ("full_wrong", full & ~fused_corr)]:
            if sel.sum() < 10:
                continue
            traj[name] = {
                "n": int(sel.sum()),
                "M_mean": prefix_M[:, sel].mean(1).tolist(),
                "M_sem": (prefix_M[:, sel].std(1) / np.sqrt(sel.sum())).tolist(),
                "H_mean": prefix_H[:, sel].mean(1).tolist(),
                "H_sem": (prefix_H[:, sel].std(1) / np.sqrt(sel.sum())).tolist(),
            }
    out["traj_groups"] = traj
    # 几何分组（按四象限）
    geo = {}
    for name in quad_names:
        m1 = name[0] == "Q"
        m10 = name[1] == "Q"
        sel = np.where((c1 == m1) & (c10 == m10))[0]
        if len(sel) < 5:
            continue
        geo[name] = {
            "n": int(len(sel)),
            "cos_gt_by_scale": cos_gt_full[:, sel].mean(1).numpy().tolist(),
            "geo_margin_by_scale": geo_margin[:, sel].mean(1).numpy().tolist(),
        }
    out["geometry_groups"] = geo

    out_dir = ROOT / "outputs" / "deep_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.dataset}_{BK}_{args.proto}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_dir / f'{args.dataset}_{BK}_{args.proto}.json'}")


if __name__ == "__main__":
    main()
