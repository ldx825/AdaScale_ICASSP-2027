#!/usr/bin/env python3
"""先验文献对照基线 + 新现象诊断（全部用缓存特征，CPU）。

方法（对每个 dataset/proto 设置）:
  B1  Margin Stop (AdapTTA-style)     : prefix margin >= delta 即退出（δ 扫描）
  B2  Selective Gating (Selective-TTA): H(single) < gamma 用单尺度，否则 full-10
  B3  Entropy Filtering (TPT/DiffTPT) : 保留最低熵 K 个 view 后融合（K 扫描）
  Ref Full-10 / Single-10 / Fixed-K
诊断:
  SNAP max-margin snapshot   : 轨迹中 margin 最大的 prefix 预测
  WGT  margin/entropy 加权融合 : w'_s = w_s * quality_s
  LOO  逐 scale 剔除融合       : 每个被剔除 scale 对融合的影响

输出: outputs/prior_baselines/<ds>_<bk>_<proto>.json
用法: python scripts/final/prior_baselines.py --dataset PET --proto text
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


def load_setting(ds, bk, proto, root):
    feats, labels, te = load_real_scales(root, ds, bk)
    if proto == "text":
        protos = te
    else:
        gf, _ = load_gen_scales(root, ds, bk, "", 10)
        protos = full_scale_aggregate(gf).mean(dim=1)
    return feats, labels.numpy(), protos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, protos = load_setting(args.dataset, args.backbone, args.proto, root)
    T = len(feats)
    N = len(labels)
    C = protos.shape[0]
    logC = float(np.log(C))
    print(f"[prior-baselines] {args.dataset}/{args.backbone}/{args.proto} N={N} C={C}")

    # ---------- Pass 1: prefix trajectory + per-scale独立量 ----------
    # prefix tensors
    prefix_sum = None
    preds = np.zeros((N, T), dtype=np.int64)
    margins = np.zeros((N, T), dtype=np.float32)
    ents = np.zeros((N, T), dtype=np.float32)
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_sum = float(SCALE_WEIGHTS[t - 1]) * f if prefix_sum is None \
            else prefix_sum + float(SCALE_WEIGHTS[t - 1]) * f
        fn = prefix_sum / prefix_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lg = fn @ protos.T
        prob = F.softmax(lg / LOGIT_TAU, dim=-1)
        t2 = torch.topk(prob, 2, dim=-1).values
        preds[:, t - 1] = lg.argmax(-1).numpy()
        margins[:, t - 1] = (t2[:, 0] - t2[:, 1]).numpy()
        ents[:, t - 1] = (-(prob * torch.log(prob.clamp_min(1e-12))).sum(-1)).numpy()

    acc_full = float((preds[:, -1] == labels).mean())
    acc_single = float((preds[:, 0] == labels).mean())

    results = {"dataset": args.dataset, "backbone": args.backbone, "proto": args.proto,
               "N": N, "C": C, "full10_acc": acc_full, "single1_acc": acc_single,
               "methods": {}, "loo": {}, "snap": {}}

    def add(method, param, acc, avg_scales):
        results["methods"].setdefault(method, []).append(
            {"param": param, "acc": acc, "avg_scales": avg_scales})

    # ---------- SNAP: max-margin snapshot ----------
    best_t = margins.argmax(axis=1)
    acc_snap = float((preds[np.arange(N), best_t] == labels).mean())
    results["snap"]["max_margin"] = {"acc": acc_snap,
                                     "avg_scale_of_pick": float((best_t + 1).mean())}
    # snap 与 final 的对比（多少样本不同）
    diff = float((preds[np.arange(N), best_t] != preds[:, -1]).mean())
    results["snap"]["frac_differs_from_final"] = diff
    print(f"  SNAP max-margin: acc={acc_snap*100:.2f}% (different from final on {diff*100:.1f}%)")

    # ---------- B1: Margin Stop ----------
    for delta in [0.2, 0.35, 0.5, 0.65, 0.8]:
        exit_t = np.full(N, T, dtype=np.int64)
        exited = np.zeros(N, dtype=bool)
        for t in range(2, T + 1):
            m = margins[:, t - 1]
            go = (~exited) & (m >= delta)
            exit_t[go] = t
            exited |= go
        pred = preds[np.arange(N), exit_t - 1]
        acc = float((pred == labels).mean())
        add("margin_stop", f"delta={delta}", acc, float(exit_t.mean()))
        print(f"  B1 margin_stop d={delta}: acc={acc*100:.2f}% scales={exit_t.mean():.2f}")

    # ---------- B2: Selective gating ----------
    # H(single)=H at scale 10 (largest scale), 相对熵
    H_single = ents[:, -1] / logC
    # single-10 独立预测
    f10 = feats[-1] / feats[-1].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    l10 = f10 @ protos.T
    p10 = F.softmax(l10 / LOGIT_TAU, dim=-1)
    pred_single10 = l10.argmax(-1).numpy()
    acc_single10 = float((pred_single10 == labels).mean())
    results["single10_independent_acc"] = acc_single10
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gamma = float(np.quantile(H_single, q))
        gate = H_single < gamma
        pred = np.where(gate, pred_single10, preds[:, -1])
        acc = float((pred == labels).mean())
        avg_scales = 1 + (1 - gate.mean()) * (T - 1)
        add("selective_gating", f"q={q}", acc, float(avg_scales))
    print(f"  B2 selective done (single10 acc={acc_single10*100:.2f}%)")

    # ---------- B3: Entropy view filtering ----------
    # 需要 per-scale 独立熵（scale 独立预测的熵）——重新计算
    scale_ents = np.zeros((N, T), dtype=np.float32)
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lg = f @ protos.T
        prob = F.softmax(lg / LOGIT_TAU, dim=-1)
        scale_ents[:, t - 1] = (-(prob * torch.log(prob.clamp_min(1e-12))).sum(-1)).numpy()
    order = np.argsort(scale_ents, axis=1)  # 熵升序
    for K in [3, 5, 7]:
        keep = order[:, :K]
        # 逐样本逐 t 累加（对每个 t 找到包含它的样本）
        fus = torch.zeros(N, feats[0].shape[1])
        for t in range(T):
            f = feats[t] / feats[t].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            in_keep = (keep == t).any(axis=1)
            if in_keep.any():
                w = float(SCALE_WEIGHTS[t])
                fus[in_keep] += f[in_keep] * w
        fus = fus / fus.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        acc = float(((fus @ protos.T).argmax(-1).numpy() == labels).mean())
        add("entropy_filter", f"K={K}", acc, float(T))
        print(f"  B3 entropy_filter K={K}: acc={acc*100:.2f}% (cost=10)")

    # ---------- WGT: quality-weighted fusion ----------
    # 每 scale 独立 margin（同 scale_ents 计算路径）
    scale_margins = np.zeros((N, T), dtype=np.float32)
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lg = f @ protos.T
        prob = F.softmax(lg / LOGIT_TAU, dim=-1)
        t2 = torch.topk(prob, 2, dim=-1).values
        scale_margins[:, t - 1] = (t2[:, 0] - t2[:, 1]).numpy()

    def weighted_fusion(weights_per_scale):  # (N,T) np
        fus = torch.zeros(N, feats[0].shape[1])
        for t in range(T):
            f = feats[t] / feats[t].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            w = torch.from_numpy(weights_per_scale[:, t].astype(np.float32))
            fus += f * w.unsqueeze(-1)
        fus = fus / fus.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return float(((fus @ protos.T).argmax(-1).numpy() == labels).mean())

    # margin 加权: w'_s = w_s * m_s（per-sample）
    W_base = np.array(SCALE_WEIGHTS[:T], dtype=np.float32)[None, :].repeat(N, 0)
    acc_wm = weighted_fusion(W_base * scale_margins)
    add("wgt_margin", "w*m", acc_wm, float(T))
    print(f"  WGT margin-weighted fusion: acc={acc_wm*100:.2f}%")

    # entropy 加权: w'_s = w_s * exp(-beta * H_s/logC)
    for beta in [0.5, 1.0, 2.0]:
        wq = np.exp(-beta * scale_ents / logC)
        acc_we = weighted_fusion(W_base * wq)
        add("wgt_entropy", f"beta={beta}", acc_we, float(T))
        print(f"  WGT entropy-weighted (beta={beta}): acc={acc_we*100:.2f}%")

    # ---------- LOO: 逐 scale 剔除 ----------
    for k in range(T):
        fus = None
        for t in range(T):
            if t == k:
                continue
            f = feats[t] / feats[t].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            fus = float(SCALE_WEIGHTS[t]) * f if fus is None else fus + float(SCALE_WEIGHTS[t]) * f
        fus = fus / fus.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        acc = float(((fus @ protos.T).argmax(-1).numpy() == labels).mean())
        results["loo"][f"drop_scale{k+1}"] = acc - acc_full  # Δ vs full
    print("  LOO:", {k: f"{v*100:+.2f}" for k, v in results["loo"].items()})

    # ---------- 保存 ----------
    out_dir = ROOT / "outputs" / "prior_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
