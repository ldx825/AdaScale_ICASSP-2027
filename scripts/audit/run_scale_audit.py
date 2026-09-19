"""
run_scale_audit.py
------------------
Scale Utility Audit 主脚本（Phase 4）。

功能:
  1. 读取官方 per-scale HDF5 缓存（real 或 gen 特征）。
  2. 验证 prefix-10 聚合与官方 *_feature_ms.hdf5 的数值一致性。
  3. 计算 per-sample / per-scale 的 prefix 预测动态。
  4. 输出 outputs/scale_audit/<dataset>/<backbone>/per_sample_<proto>.parquet
     和 summary_<proto>.json。

用法:
  python scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/32 --proto text
"""
import os
import sys
import json
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.load_features import (
    load_real_scales, load_gen_scales, prefix_aggregate, full_scale_aggregate,
    real_ms_path, gen_ms_path,
)
from src.analysis.prefix_analysis import (
    prefix_logits, softmax_stats, js_divergence,
    min_final_equivalent_scale, flip_statistics, harmful_rescue,
    first_last_correct_scale, prefix_accuracy, oracle_min_scales,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PET")
    p.add_argument("--backbone", default="ViT-B/32")
    p.add_argument("--proto", default="text", choices=["text", "gen"],
                   help="分类头: text=文本原型(vanilla 协议), gen=生成图原型(LG-CLIP 协议)")
    p.add_argument("--llm", default="", help="生成图前缀 LLM_ 或空")
    p.add_argument("--ngen", default=10, type=int)
    p.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--out_root", default=os.path.join(PROJECT_ROOT, "outputs", "scale_audit"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_samples", default=0, type=int, help="调试用: 限制样本数")
    return p.parse_args()


def verify_prefix10(feats, ms_path, device):
    """验证 prefix-10 聚合 == 官方 ms 文件（数值一致性）。"""
    f10 = full_scale_aggregate([f.to(device) for f in feats])
    with h5py.File(ms_path, "r") as f:
        official = torch.from_numpy(np.array(f["test_f"])).float().to(device)
    diff = (f10 - official).abs().max().item()
    return diff


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_root, args.dataset,
                             args.backbone.replace("/", "")), exist_ok=True)
    tag = args.backbone.replace("/", "")
    out_dir = os.path.join(args.out_root, args.dataset, tag)

    print(f"[audit] dataset={args.dataset} backbone={args.backbone} proto={args.proto}")
    # ---- 加载 real 特征 ----
    feats, labels, text_emb = load_real_scales(args.image_root, args.dataset, args.backbone)
    N = feats[0].shape[0]
    if args.max_samples:
        N = min(N, args.max_samples)
        feats = [f[:N] for f in feats]
        labels = labels[:N]
    print(f"  loaded {len(feats)} scales, N={N}, D={feats[0].shape[1]}")

    # ---- 一致性验证 ----
    ms_path = real_ms_path(args.image_root, args.dataset, args.backbone)
    if os.path.exists(ms_path):
        diff = verify_prefix10(feats, ms_path, "cpu")
        print(f"  [verify] prefix-10 vs official ms: max|diff| = {diff:.3e}")
        assert diff < 1e-4, f"MS 融合不一致: {diff}"
    else:
        diff = None
        print(f"  [verify] official ms file missing ({ms_path}) -> verification skipped "
              f"(non-default backbone)")

    # ---- prototypes ----
    if args.proto == "text":
        prototypes = text_emb
        proto_note = "text prototypes (all_embeddings)"
    else:
        gen_feats, _ = load_gen_scales(args.image_root, args.dataset, args.backbone,
                                       args.llm, args.ngen)
        gf = full_scale_aggregate([g.to(args.device) for g in gen_feats])
        prototypes = gf.mean(dim=1)  # (C, D)，与 mega.py 一致（不重新 normalize）
        proto_note = f"gen prototypes (llm={args.llm!r}, Ngen={args.ngen})"
    print(f"  prototypes: {tuple(prototypes.shape)} ({proto_note})")

    # ---- prefix logits ----
    logits, feats_prefix = prefix_logits(feats, prototypes, device=args.device)
    stats = softmax_stats(logits)
    preds = stats["top1"].cpu().numpy()          # (T, N)
    preds = preds.T                              # (N, T)
    labels_np = labels.numpy()
    T = preds.shape[1]

    # ---- 统计 ----
    acc_curve = prefix_accuracy(preds, labels_np)
    flips = flip_statistics(preds)
    teq = min_final_equivalent_scale(preds)
    teq_gt, t_gt_first = oracle_min_scales(labels_np, preds)
    hr = harmful_rescue(labels_np, preds)
    first_c, last_c = first_last_correct_scale(labels_np, preds)

    # per-scale 独立 acc（对应官方脚本打印）
    ind_acc = []
    for t in range(T):
        sims = feats[t].to(args.device) @ prototypes.to(args.device).T
        ind_acc.append((sims.argmax(-1).cpu().numpy() == labels_np).mean())

    # JS divergence to prev / feature cosines
    p = stats["p"]                                # (T, N, C)
    js_prev = np.zeros((T, N))
    for t in range(1, T):
        js_prev[t] = js_divergence(p[t], p[t - 1]).cpu().numpy()
    fcos_prev = np.zeros((T, N))
    fcos_s1 = np.zeros((T, N))
    for t in range(T):
        fp = feats_prefix[t]
        fcos_s1[t] = (fp * feats_prefix[0]).sum(-1).cpu().numpy()
        if t > 0:
            fcos_prev[t] = (fp * feats_prefix[t - 1]).sum(-1).cpu().numpy()

    # logit l2 change
    logit_l2 = np.zeros((T, N))
    for t in range(1, T):
        logit_l2[t] = (logits[t] - logits[t - 1]).norm(dim=-1).cpu().numpy()

    # ---- 构造 per-sample 表 ----
    rows = []
    full10_top1 = preds[:, -1]
    full10_correct = full10_top1 == labels_np
    for t in range(T):
        for i in range(N):
            changed = (t > 0) and (preds[i, t] != preds[i, t - 1])
            rows.append((
                i, t + 1,
                int(labels_np[i]),
                int(preds[i, t]),
                float(stats["top1_raw"][t, i]),
                int(stats["top1"][t, i]),
                float(stats["top1_prob"][t, i]),
                float(stats["top2_prob"][t, i]),
                float(stats["margin"][t, i]),
                float(stats["raw_margin"][t, i]),
                float(stats["entropy"][t, i]),
                bool(preds[i, t] == labels_np[i]),
                int(full10_top1[i]),
                bool(full10_correct[i]),
                bool(changed),
                float(logit_l2[t, i]),
                float(fcos_prev[t, i]),
                float(fcos_s1[t, i]),
                float(js_prev[t, i]),
            ))
    cols = ["sample_id", "scale_id", "ground_truth", "top1_class",
            "top1_raw", "top1_prob_class", "top1_prob", "top2_prob", "margin",
            "raw_margin", "entropy", "correct", "full10_top1", "full10_correct",
            "prediction_changed_from_prev", "logit_l2_change",
            "feature_cosine_to_prev", "feature_cosine_to_scale1",
            "js_divergence_to_prev"]
    df = pd.DataFrame(rows, columns=cols)
    pq_path = os.path.join(out_dir, f"per_sample_{args.proto}.parquet")
    df.to_parquet(pq_path, index=False)
    print(f"  saved per-sample: {pq_path} ({len(df)} rows)")

    # ---- summary ----
    summary = OrderedDict()
    summary["dataset"] = args.dataset
    summary["backbone"] = args.backbone
    summary["prototype"] = proto_note
    summary["num_samples"] = int(N)
    summary["num_scales"] = int(T)
    summary["verify_prefix10_max_diff"] = float(diff) if diff is not None else None
    summary["prefix_acc_curve"] = [float(a) for a in acc_curve]
    summary["per_scale_independent_acc"] = [float(a) for a in ind_acc]
    summary["prefix_marginal_gain"] = [float(acc_curve[0])] + [
        float(acc_curve[t] - acc_curve[t - 1]) for t in range(1, T)]
    summary["flip_rate_per_transition"] = [float(x) for x in flips["flip_rate"]]
    summary["mean_flips_per_sample"] = float(flips["n_flips"].mean())
    summary["flip_count_hist"] = {int(k): int(v) for k, v in
                                  zip(*np.unique(flips["n_flips"], return_counts=True))}
    # t_eq
    teq_hist = np.bincount(teq, minlength=T + 1)[1:]
    summary["teq_hist"] = {int(i + 1): int(c) for i, c in enumerate(teq_hist)}
    summary["teq_cdf"] = [float(x) for x in np.cumsum(np.bincount(teq, minlength=T + 1)[1:]) / N]
    summary["teq_mean"] = float(teq.mean())
    summary["teq_median"] = float(np.median(teq))
    summary["teq_leq"] = {k: float((teq <= k).mean()) for k in range(1, T + 1)}
    # harmful / rescue
    summary["harmful_rate"] = float(hr["harmful"].mean())
    summary["rescue_rate"] = float(hr["rescue"].mean())
    summary["harmful_transitions"] = [int(x) for x in hr["harmful_transition"]]
    summary["rescue_transitions"] = [int(x) for x in hr["rescue_transition"]]
    summary["full10_acc"] = float(acc_curve[-1])
    # per-sample aggregate signals
    summary["margin_final_quantiles"] = {
        str(q): float(np.quantile(stats["margin"][-1].cpu().numpy(), q))
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
    # stable-vs-unstable margin comparison（用 t=1..2 的信号预测后续稳定性，label-free 场景）
    stable_early = teq <= 3
    summary["teq_leq3_rate"] = float(stable_early.mean())
    with open(os.path.join(out_dir, f"summary_{args.proto}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"  saved summary: {os.path.join(out_dir, f'summary_{args.proto}.json')}")
    print(f"  full10 acc = {acc_curve[-1]*100:.2f}% |  Acc(1)={acc_curve[0]*100:.2f}%"
          f" | teq median={np.median(teq):.0f} mean={teq.mean():.2f} | harmful={hr['harmful'].mean()*100:.2f}%"
          f" rescue={hr['rescue'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
