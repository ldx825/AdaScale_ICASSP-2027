#!/usr/bin/env python3
"""Seed replication：验证主结论对 crop 随机种子的稳健性。

对 `--out_tag` 后缀的 per-scale 特征（不同 seed 提取）：
  1) prefix 曲线（形状/峰值/末段趋势）
  2) 融合 vs 单尺度对比
  3) AdaScale v2 完整评估（模式选择 + 早停）
并与主结果（seed=2024）对比。

用法:
  # 先提取（GPU）： python src/analysis/extract_ms_batched.py --dataset PET \
  #   --backbone ViT-B/32 --mode real --seed 1234 --out_tag _seed1234
  python scripts/final/seed_replication.py --dataset PET --proto text --tag _seed1234
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
    real_scale_path, gen_scale_path, load_gen_scales, full_scale_aggregate,
    SCALE_WEIGHTS, NUM_SCALES,
)
from src.adascale.simulate import StabilityPolicy, simulate  # noqa: E402

import h5py  # noqa: E402

LOGIT_TAU = 0.01
STAB = (4, 0.45, 0.01)
THETA = 0.01
CAL_N = 500


def load_tagged_real(image_root, dataset, backbone, tag, dtype=torch.float32):
    feats, labels, text_emb = [], None, None
    for s in range(1, NUM_SCALES + 1):
        p = real_scale_path(image_root, dataset, backbone, s)
        p = p.replace(".hdf5", f"{tag}.hdf5")
        with h5py.File(p, "r") as f:
            feats.append(torch.from_numpy(np.array(f["test_f"])).to(dtype))
            if labels is None:
                labels = torch.from_numpy(np.array(f["test_l"]))
                text_emb = torch.from_numpy(np.array(f["all_embeddings"])).to(dtype)
    return feats, labels, text_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--tag", required=True, help="e.g. _seed1234")
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, text_emb = load_tagged_real(root, args.dataset, args.backbone, args.tag)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gf, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gf).mean(dim=1)
    labels = labels.numpy()
    N = len(labels)

    # prefix 曲线
    prefix_acc = []
    ps = None
    for i, f in enumerate(feats):
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ps = SCALE_WEIGHTS[i] * f if ps is None else ps + SCALE_WEIGHTS[i] * f
        fn = ps / ps.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prefix_acc.append(float(((fn @ prototypes.T).argmax(-1).numpy() == labels).mean()))

    # 融合 vs 单尺度
    fn = ps / ps.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fs = feats[-1] / feats[-1].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lf, ls = fn @ prototypes.T, fs @ prototypes.T
    acc_f = float((lf.argmax(-1).numpy() == labels).mean())
    acc_s = float((ls.argmax(-1).numpy() == labels).mean())

    pf = F.softmax(lf / LOGIT_TAU, -1)
    pss = F.softmax(ls / LOGIT_TAU, -1)
    Hf = -(pf * torch.log(pf.clamp_min(1e-12))).sum(-1).numpy()
    Hs = -(pss * torch.log(pss.clamp_min(1e-12))).sum(-1).numpy()
    C = prototypes.shape[0]
    rng = np.random.default_rng(0)
    cal = rng.choice(N, size=min(CAL_N, N), replace=False)
    rel_diff = float((Hf[cal] - Hs[cal]).mean() / np.log(C))
    mode = "SINGLE" if rel_diff > THETA else "FUSED"

    if mode == "SINGLE":
        acc_v2, sc_v2 = acc_s, 1.0
    else:
        pol = StabilityPolicy(r=STAB[0], delta=STAB[1], eps=STAB[2],
                              min_scale=1, use_js=True, use_margin=True)
        res = simulate(feats, prototypes, pol)
        acc_v2 = float((res.pred == labels).mean())
        sc_v2 = float(res.scales_used.mean())

    out = {
        "dataset": args.dataset, "proto": args.proto, "tag": args.tag, "N": N,
        "prefix_acc_curve": prefix_acc,
        "acc_fused10": acc_f, "acc_single10": acc_s,
        "rel_entropy_diff_cal": rel_diff, "mode": mode,
        "adascale_acc": acc_v2, "adascale_scales": sc_v2,
    }
    out_path = ROOT / "outputs" / "final" / \
        f"seed_replication_{args.dataset}_{args.proto}{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[seed-repl] {args.dataset}/{args.proto}{args.tag} N={N}")
    print(f"  prefix: {[round(a*100,1) for a in prefix_acc]}")
    print(f"  fused={acc_f*100:.2f}% single={acc_s*100:.2f}% "
          f"relH={rel_diff:+.5f} mode={mode}")
    print(f"  AdaScale=v2 {acc_v2*100:.2f}% @ {sc_v2:.2f} scales")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
