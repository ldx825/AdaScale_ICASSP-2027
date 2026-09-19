#!/usr/bin/env python3
"""单尺度与选择型基线分析（为 EUROSAT 现象服务）。

动机：EUROSAT 上 10-scale 三角加权融合 (42.38%) < 单尺度 scale-10 (45.35%)，
说明"融合本身"可能有害。本脚本系统对比：
  - last-only:      只用最大 scale（10）的预测
  - first-only:     只用最小 scale（1）
  - max-margin:     逐样本选择 margin 最大的单 scale 预测（无需 GT）
  - max-logit:      逐样本选择 top1 logit 最大的单 scale 预测
  - oracle-scale:   逐样本选择正确的单 scale（上界，仅分析用，禁止入主表）
  - msvote:         单 scale 多数投票
对所有数据集/协议输出 acc。

用法:
    python scripts/final/single_scale_baselines.py --dataset EUROSAT --proto text
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
    load_real_scales, load_gen_scales, full_scale_aggregate,
)

LOGIT_TAU = 0.01


def get_feats_protos(dataset, backbone, proto, image_root, ngen=10):
    feats, labels, text_emb = load_real_scales(str(image_root), dataset, backbone)
    if proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(str(image_root), dataset, backbone, "", ngen)
        prototypes = full_scale_aggregate(gen_feats).mean(dim=1)
    return feats, labels.numpy(), prototypes


def per_scale_logits_probs(feats, prototypes):
    """返回 (logits[N,T,C], probs[N,T,C], preds[N,T])。"""
    N = feats[0].shape[0]
    T = len(feats)
    C = prototypes.shape[0]
    logits_all = np.zeros((N, T, C), dtype=np.float32)
    for t in range(T):
        f = feats[t] / feats[t].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits_all[:, t, :] = (f @ prototypes.T).numpy()
    probs = F.softmax(torch.from_numpy(logits_all) / LOGIT_TAU, dim=-1).numpy()
    preds = logits_all.argmax(-1)
    return logits_all, probs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--ngen", type=int, default=10)
    ap.add_argument("--image_root", default=str(ROOT / "data"))
    args = ap.parse_args()

    feats, labels, prototypes = get_feats_protos(
        args.dataset, args.backbone, args.proto, args.image_root, args.ngen)
    logits, probs, preds = per_scale_logits_probs(feats, prototypes)
    N, T, C = probs.shape
    print(f"[baselines] {args.dataset}/{args.backbone}/{args.proto} N={N} T={T}")

    results = {}
    # last-only / first-only
    results["first-only(s1)"] = float((preds[:, 0] == labels).mean())
    results["last-only(s10)"] = float((preds[:, -1] == labels).mean())
    for t in range(T):
        results[f"single(s{t+1})"] = float((preds[:, t] == labels).mean())

    # max-margin selection
    top2 = np.sort(probs, axis=-1)[:, :, -2:]
    margins = top2[:, :, 1] - top2[:, :, 0]
    sel = margins.argmax(axis=1)
    results["max-margin"] = float((preds[np.arange(N), sel] == labels).mean())

    # max-logit selection (top-1 logit 最大)
    sel2 = logits.max(-1).argmax(axis=1)
    results["max-logit"] = float((preds[np.arange(N), sel2] == labels).mean())

    # oracle per-sample best scale
    correct = (preds == labels[:, None]).astype(np.float32)
    results["oracle-best-single"] = float(correct.max(axis=1).mean())

    # majority vote
    from scipy import stats as _s
    votes = _s.mode(preds, axis=1, keepdims=False).mode
    results["msvote"] = float((votes == labels).mean())

    for k, v in results.items():
        if not k.startswith("single(s"):
            print(f"  {k:22s} {v*100:6.2f}%")

    out = ROOT / "outputs" / "single_scale" / f"{args.dataset}_{args.backbone.replace('/','')}_{args.proto}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
