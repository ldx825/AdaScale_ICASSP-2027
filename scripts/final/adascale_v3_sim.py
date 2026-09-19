#!/usr/bin/env python3
"""AdaScale v3 离线模拟：冻结 v2 退出轨迹，仅改进融合输出规则。

在缓存特征上模拟（无 GPU 前向）：
  v2 输出：prefix 三角加权平均（w_s = s）
  v3 候选：
    A) margin 加权:  w_s = s * margin_s^beta （因果：每视图自身统计）
    B) 熵过滤:      丢弃已视图中熵最高的 K 个（仅对走满样本全可见时最可靠）
    C) A+B 组合
    E) 保守版:      仅对走满（t=10）样本替换输出为过滤/加权融合，早退样本保持 v2

用法:
  python scripts/final/adascale_v3_sim.py --dataset PET --proto text
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
        gen_feats, _ = load_gen_scales(root, dataset, backbone, "", 10)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    return feats, labels.numpy(), prototypes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--stab", default="4,0.65,0.01")
    args = ap.parse_args()
    r_s, d_s, e_s = [float(x) for x in args.stab.split(",")]
    stab_tag = f"r{int(r_s)}_d{d_s}_e{e_s}"

    feats, labels, protos = load_all(args.dataset, args.backbone, args.proto)
    N = len(labels)
    T = len(feats)
    BK = args.backbone.replace("/", "")
    npz_path = ROOT / "outputs" / "final" / \
        f"adascale_v2_{args.dataset}_{BK}_{args.proto}_{stab_tag}_preds.npz"
    z = np.load(npz_path)
    scales_v2 = z["scales_v2"]          # (N,) 每样本退出时使用的视图数
    pred_v2 = z["pred_v2"]
    print(f"[v3-sim] {args.dataset}/{args.backbone}/{args.proto} N={N} "
          f"acc_v2={(pred_v2 == labels).mean():.4f}")
    print(f"  exit hist: {np.bincount(scales_v2, minlength=T + 1)[1:].tolist()}")

    # ---- 全视图矩阵 ----
    F_all = torch.stack([f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                         for f in feats])                      # (T,N,D)
    logits_all = torch.einsum("tnd,cd->tnc", F_all, protos)    # 未除 tau
    P_all = F.softmax(logits_all / LOGIT_TAU, dim=-1)          # (T,N,C)
    H_all = -(P_all * torch.log(P_all.clamp_min(1e-12))).sum(-1)   # (T,N)
    top2 = torch.topk(P_all, 2, dim=-1).values
    M_all = (top2[..., 0] - top2[..., 1])                      # (T,N)
    preds_single = logits_all.argmax(-1)                       # (T,N) 各视图预测

    idx_np = np.arange(N)
    t_i = scales_v2
    # mask[t,i] = (t < t_i) 即样本 i 的已见视图
    mask = (np.arange(T)[:, None] < t_i[None, :]).astype(np.float32)  # (T,N)

    W_tri = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * mask      # 三角权重

    def fuse_from_weights(W):
        """W: (T,N) 权重 -> 预测 (N,)"""
        Wt = torch.from_numpy(W)
        fs = torch.einsum("tn,tnd->nd", Wt, F_all)
        fs = fs / fs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (fs @ protos.T).argmax(-1).numpy()

    results = {}
    acc_v2 = float((pred_v2 == labels).mean())
    results["v2_recompute"] = float((fuse_from_weights(W_tri) == labels).mean())

    # 方案 A: margin 加权
    for beta in (0.5, 1.0, 2.0):
        W = W_tri * (M_all.numpy() ** beta)
        results[f"A_margin_beta{beta}"] = float((fuse_from_weights(W) == labels).mean())

    # 方案 B: 熵过滤（丢弃已见视图中熵最高的 K 个）
    for K in (3, 5):
        keep = np.maximum(t_i - K, 1)          # 至少保留 1 个视图
        order = np.argsort(H_all.numpy(), axis=0)      # (T,N) 熵升序
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(T)[:, None].repeat(N, 1), axis=0)
        keep_mask = (rank < keep[None, :]).astype(np.float32) * mask
        W = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * keep_mask
        results[f"B_entropy_filter_K{K}"] = float((fuse_from_weights(W) == labels).mean())

    # 方案 C: B(K=3) + margin beta=1
    K = 3
    keep = np.maximum(t_i - K, 1)
    order = np.argsort(H_all.numpy(), axis=0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(T)[:, None].repeat(N, 1), axis=0)
    keep_mask = (rank < keep[None, :]).astype(np.float32) * mask
    W = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * keep_mask * M_all.numpy()
    results["C_filterK3_margin1"] = float((fuse_from_weights(W) == labels).mean())

    # 方案 E: 仅对走满样本 (t=10) 替换为过滤/加权融合；早退样本保持 v2
    full = (t_i == T)
    W = W_tri.copy()
    keep = np.maximum(t_i - 5, 1)
    keep_mask = (rank < keep[None, :]).astype(np.float32) * mask
    W_e = W.copy()
    W_e[:, full] = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * keep_mask)[:, full]
    results["E_fullonly_filterK5"] = float((fuse_from_weights(W_e) == labels).mean())

    # ---- E-sweep: 走满样本规则扫描（K 过滤 / margin 加权 / 组合 / 早退加权）----
    print("\n  == full-sample rule sweep (走满样本) ==")
    Mn = M_all.numpy()
    for K in (0, 1, 2, 3, 4, 5, 6, 7):
        keepK = np.maximum(np.full(N, T) - K, 1)
        kmask = (rank < keepK[None, :]).astype(np.float32)
        Wf = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * kmask
        W2 = W_tri.copy(); W2[:, full] = Wf[:, full]
        acc = float((fuse_from_weights(W2) == labels).mean())
        results[f"E_K{K}"] = acc
        print(f"  E_K{K:<2d} (full={T})                acc={acc:.4f}  ({acc - acc_v2:+.4f})")
    for beta in (1.0, 2.0):
        Wf = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * (Mn ** beta)
        W2 = W_tri.copy(); W2[:, full] = Wf[:, full]
        acc = float((fuse_from_weights(W2) == labels).mean())
        results[f"E_margin_beta{beta}"] = acc
        print(f"  E_margin_beta{beta}             acc={acc:.4f}  ({acc - acc_v2:+.4f})")
    # 组合: K=4 过滤 + margin^1
    keepK = np.maximum(np.full(N, T) - 4, 1)
    kmask = (rank < keepK[None, :]).astype(np.float32)
    Wf = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * kmask * Mn
    W2 = W_tri.copy(); W2[:, full] = Wf[:, full]
    acc = float((fuse_from_weights(W2) == labels).mean())
    results["E_K4_margin1"] = acc
    print(f"  E_K4_margin1                acc={acc:.4f}  ({acc - acc_v2:+.4f})")
    # 早退样本 margin 加权（所有样本加权，不做过滤）
    W2 = W_tri * (Mn ** 1.0)
    acc = float((fuse_from_weights(W2) == labels).mean())
    results["A_all_margin1"] = acc
    print(f"  A_all_margin1 (全局)         acc={acc:.4f}  ({acc - acc_v2:+.4f})")
    # 早退保持,走满 margin^1（组合）
    W2 = W_tri * (Mn ** 1.0)
    W3 = W_tri.copy()
    W3[:, full] = W2[:, full]
    acc = float((fuse_from_weights(W3) == labels).mean())
    results["E_margin1_only"] = acc
    print(f"  E_margin1_only              acc={acc:.4f}  ({acc - acc_v2:+.4f})")

    # ---- 方案 F: 概率层融合（集成风格，替代特征层平均）----
    print("\n  == F: probability-level fusion ==")
    Pn = P_all.numpy()
    Wp = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * mask)[:, :, None]  # (T,N,1)
    Wm = mask[:, :, None]
    # F1: 三角加权概率平均
    P_f1 = (Pn * Wp).sum(0) / Wp.sum(0).clip(min=1e-9)
    acc_f1 = float((P_f1.argmax(1) == labels).mean())
    results["F1_prob_tri_mean"] = acc_f1
    print(f"  F1_prob_tri_mean            acc={acc_f1:.4f}  ({acc_f1 - acc_v2:+.4f})")
    # F2: 均匀概率平均
    P_f2 = (Pn * Wm).sum(0) / Wm.sum(0).clip(min=1e-9)
    acc_f2 = float((P_f2.argmax(1) == labels).mean())
    results["F2_prob_mean"] = acc_f2
    print(f"  F2_prob_mean                acc={acc_f2:.4f}  ({acc_f2 - acc_v2:+.4f})")
    # F3: 三角加权几何平均（对数概率平均）
    logP = np.log(np.clip(Pn, 1e-12, None))
    lg_f3 = (logP * Wp).sum(0) / Wp.sum(0).clip(min=1e-9)
    acc_f3 = float((lg_f3.argmax(1) == labels).mean())
    results["F3_prob_geo_tri"] = acc_f3
    print(f"  F3_prob_geo_tri             acc={acc_f3:.4f}  ({acc_f3 - acc_v2:+.4f})")
    # F4: 三角加权概率平均 - 仅走满样本替换（早退样本保持 v2 特征融合输出）
    fs_v2 = torch.einsum("tn,tnd->nd", torch.from_numpy(W_tri), F_all)
    fs_v2 = fs_v2 / fs_v2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    P_v2 = F.softmax((fs_v2 @ protos.T) / LOGIT_TAU, dim=-1).numpy()
    P_f4 = P_v2.copy()
    P_f4[full] = P_f1[full]
    acc_f4 = float((P_f4.argmax(1) == labels).mean())
    results["F4_prob_tri_fullonly"] = acc_f4
    print(f"  F4_prob_tri_fullonly        acc={acc_f4:.4f}  ({acc_f4 - acc_v2:+.4f})")
    # F5: 数值稳定 logsumexp 融合（特征+概率混合：logit 平均）
    lg_all = logits_all.numpy() / LOGIT_TAU
    lg_f5 = (lg_all * Wm).sum(0) / Wm.sum(0).clip(min=1e-9)
    acc_f5 = float((lg_f5.argmax(1) == labels).mean())
    results["F5_logit_mean"] = acc_f5
    print(f"  F5_logit_mean               acc={acc_f5:.4f}  ({acc_f5 - acc_v2:+.4f})")
    # F6: F1 与 K=5 过滤组合（概率域）
    keep5b = np.maximum(t_i - 5, 1)
    km5b = (rank < keep5b[None, :]).astype(np.float32)
    Wp6 = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * km5b)[:, :, None]
    P_f6 = (Pn * Wp6).sum(0) / Wp6.sum(0).clip(min=1e-9)
    acc_f6 = float((P_f6.argmax(1) == labels).mean())
    results["F6_prob_tri_filtK5"] = acc_f6
    print(f"  F6_prob_tri_filtK5          acc={acc_f6:.4f}  ({acc_f6 - acc_v2:+.4f})")

    # F7: 不同温度下的三角加权概率平均
    for tau2 in (0.02, 0.05):
        P2 = F.softmax(logits_all / tau2, dim=-1).numpy()
        P_f7 = (P2 * Wp).sum(0) / Wp.sum(0).clip(min=1e-9)
        acc_f7 = float((P_f7.argmax(1) == labels).mean())
        results[f"F7_prob_tri_tau{tau2}"] = acc_f7
        print(f"  F7_prob_tri_tau{tau2:<4}          acc={acc_f7:.4f}  ({acc_f7 - acc_v2:+.4f})")
    # F8: 概率三角 + margin 加权 (β=0.5, 1)
    for beta in (0.5, 1.0):
        Wp8 = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * mask *
               (Mn ** beta))[:, :, None]
        P_f8 = (Pn * Wp8).sum(0) / Wp8.sum(0).clip(min=1e-9)
        acc_f8 = float((P_f8.argmax(1) == labels).mean())
        results[f"F8_prob_tri_margin{beta}"] = acc_f8
        print(f"  F8_prob_tri_margin{beta:<4}       acc={acc_f8:.4f}  ({acc_f8 - acc_v2:+.4f})")
    # F9: v2 特征融合 vs F1 概率平均的置信度选择（取 margin 更大者）
    def _margin(p):
        t = torch.from_numpy(p)
        v = t.topk(2, dim=-1).values
        return (v[:, 0] - v[:, 1]).numpy()
    m_v2 = _margin(P_v2)
    m_f1 = _margin(P_f1)
    pred_sel = np.where(m_f1 >= m_v2, P_f1.argmax(1), P_v2.argmax(1))
    acc_f9 = float((pred_sel == labels).mean())
    results["F9_conf_select"] = acc_f9
    print(f"  F9_conf_select              acc={acc_f9:.4f}  ({acc_f9 - acc_v2:+.4f})")
    # F10: τ=0.02 概率三角平均 + margin^0.5
    P2 = F.softmax(logits_all / 0.02, dim=-1).numpy()
    Wp10 = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * mask *
            (Mn ** 0.5))[:, :, None]
    P_f10 = (P2 * Wp10).sum(0) / Wp10.sum(0).clip(min=1e-9)
    acc_f10 = float((P_f10.argmax(1) == labels).mean())
    results["F10_prob_tau02_margin05"] = acc_f10
    print(f"  F10_prob_tau02_margin05     acc={acc_f10:.4f}  ({acc_f10 - acc_v2:+.4f})")
    # F11: τ=0.02 概率均匀平均 + margin^0.5
    Wp11 = (mask * (Mn ** 0.5))[:, :, None]
    P_f11 = (P2 * Wp11).sum(0) / Wp11.sum(0).clip(min=1e-9)
    acc_f11 = float((P_f11.argmax(1) == labels).mean())
    results["F11_prob_flat_margin05"] = acc_f11
    print(f"  F11_prob_flat_margin05      acc={acc_f11:.4f}  ({acc_f11 - acc_v2:+.4f})")

    # ---- 配对 bootstrap（最佳候选 vs v2）----
    rng = np.random.default_rng(0)
    B = 1000
    cand = {"F7": results["F7_prob_tri_tau0.02"],
            "F8m05": results["F8_prob_tri_margin0.5"],
            "F10": results["F10_prob_tau02_margin05"]}
    # 重算候选预测
    pred_map = {}
    P_f7b = (F.softmax(logits_all / 0.02, dim=-1).numpy() * Wp).sum(0) / Wp.sum(0).clip(min=1e-9)
    pred_map["F7"] = P_f7b.argmax(1)
    Wp8b = (SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * mask * (Mn ** 0.5))[:, :, None]
    pred_map["F8m05"] = ((Pn * Wp8b).sum(0) / Wp8b.sum(0).clip(min=1e-9)).argmax(1)
    pred_map["F10"] = P_f10.argmax(1)
    correct_v2 = (pred_v2 == labels)
    print("\n  == paired bootstrap vs v2 (B=1000) ==")
    for name, pm in pred_map.items():
        c_new = (pm == labels)
        deltas = []
        for _ in range(B):
            ii = rng.integers(0, N, N)
            deltas.append(c_new[ii].mean() - correct_v2[ii].mean())
        deltas = np.array(deltas)
        print(f"  {name:6s} dacc={np.mean(deltas):+.4f} "
              f"[{np.percentile(deltas, 2.5):+.4f},{np.percentile(deltas, 97.5):+.4f}] "
              f"P(d>0)={np.mean(deltas > 0):.3f}")
        results[f"boot_{name}_ci"] = [float(np.percentile(deltas, 2.5)),
                                      float(np.percentile(deltas, 97.5)),
                                      float(np.mean(deltas > 0))]

    # ---- 分组诊断 ----
    early = ~full
    print(f"\n  early-exit: {early.sum()} (acc {(pred_v2[early] == labels[early]).mean():.4f}) | "
          f"full: {full.sum()} (acc {(pred_v2[full] == labels[full]).mean():.4f})")

    # 走满样本：K=5 过滤的翻转分析
    if full.sum() > 0:
        keep5 = np.maximum(np.full(N, T) - 5, 1)
        km5 = (rank < keep5[None, :]).astype(np.float32)
        Wk5 = SCALE_WEIGHTS[:T].astype(np.float32)[:, None] * km5
        Wf5 = W_tri.copy(); Wf5[:, full] = Wk5[:, full]
        pred_f5 = fuse_from_weights(Wf5)
        r2w = ((pred_v2 == labels) & (pred_f5 != labels) & full).sum()
        w2r = ((pred_v2 != labels) & (pred_f5 == labels) & full).sum()
        print(f"  full-sample K=5 flips: wrong->right {w2r}, right->wrong {r2w}")
        # 熵结构: 正确 vs 错误走满样本的平均熵（全视图 & 最低5视图）
        Hn = H_all.numpy()
        ok_f = (pred_v2 == labels) & full
        bad_f = (pred_v2 != labels) & full
        h_all_ok = Hn[:, ok_f].mean(); h_all_bad = Hn[:, bad_f].mean()
        h_low5_ok = np.sort(Hn[:, ok_f], axis=0)[:5].mean()
        h_low5_bad = np.sort(Hn[:, bad_f], axis=0)[:5].mean()
        print(f"  mean H: ok-full {h_all_ok:.3f} (low5 {h_low5_ok:.3f}) | "
              f"bad-full {h_all_bad:.3f} (low5 {h_low5_bad:.3f})")
        # 关键区分量: mean(top10 H) - mean(low5 H) 对 acc 的预兆
        gap_ok = (Hn[:, ok_f].mean(0) - np.sort(Hn[:, ok_f], axis=0)[:5].mean(0))
        gap_bad = (Hn[:, bad_f].mean(0) - np.sort(Hn[:, bad_f], axis=0)[:5].mean(0))
        print(f"  H gap (all-low5): ok {gap_ok.mean():.3f} vs bad {gap_bad.mean():.3f}")
    # v2 acc 校验
    acc_v2 = float((pred_v2 == labels).mean())
    print(f"  v2 acc={acc_v2:.4f} | recompute={results['v2_recompute']:.4f} "
          f"(delta {results['v2_recompute'] - acc_v2:+.4f})")

    print("\n  == v3 candidates ==")
    for k, v in results.items():
        if k == "v2_recompute" or not isinstance(v, float):
            continue
        print(f"  {k:24s} acc={v:.4f}  ({v - acc_v2:+.4f})")

    out = {"dataset": args.dataset, "backbone": args.backbone, "proto": args.proto,
           "N": int(N), "acc_v2": acc_v2, "results": results,
           "n_early": int(early.sum()), "n_full": int(full.sum())}
    out_dir = ROOT / "outputs" / "final" / "v3_sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.dataset}_{BK}_{args.proto}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_dir / f'{args.dataset}_{BK}_{args.proto}.json'}")

    # 保存 v3 (F7) per-sample 预测 → 供 bootstrap / collect 下游使用
    np.savez(ROOT / "outputs" / "final" /
             f"adascale_v3_{args.dataset}_{BK}_{args.proto}_{stab_tag}_preds.npz",
             labels=labels, pred_v3=pred_map["F7"], scales_v2=scales_v2)
    print(f"[saved] adascale_v3_{args.dataset}_{BK}_{args.proto}_{stab_tag}_preds.npz")


if __name__ == "__main__":
    main()
