#!/usr/bin/env python3
"""Attention 定量统计：跨尺度 CLIP attention rollout 的集中度 vs 正确性。

对采样样本 × 10 尺度：
  - 用与提取管线一致 RRC crop + seed
  - capture attention -> rollout 7x7 图
  - 指标: 归一化 attention 熵 H/log(49)（1=均匀分散, 0=完全集中）、
           top-5 patch 质量占比、质心离中心距离
  - 同 crop 的单视图预测正确性

输出: outputs/attn_stats/<DS>_<proto>.json  + figures/deep_diag/figA_attn_stats.{png,pdf}

用法: python scripts/analysis/attn_stats.py --dataset EUROSAT --proto text --n 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "third_party" / "LG-CLIP"))

import clip  # noqa: E402
from attention_maps import patch_attention, rollout_attn, make_crop, _CAPTURED  # noqa: E402

from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate,
)

LOGIT_TAU = 0.01
SCALES = list(range(1, 11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, args.dataset, args.backbone)
    labels = labels.numpy()
    if args.proto == "text":
        protos = text_emb
    else:
        gf, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        protos = full_scale_aggregate(gf).mean(dim=1)

    if args.dataset == "EUROSAT":
        split = json.load(open("data/EUROSAT/split_EuroSAT.json"))
        paths = [str(Path("data/EUROSAT/2750") / e[0]) for e in split["test"]]
    elif args.dataset == "PET":
        split = json.load(open("data/PET/OxfordPets/split_OxfordPets.json"))
        paths = [str(Path("data/PET/OxfordPets/images") / e[0]) for e in split["test"]]
    else:
        raise ValueError("add dataset")

    N = len(labels)
    rng = np.random.default_rng(args.seed)
    sample_idx = np.sort(rng.choice(N, size=min(args.n, N), replace=False))
    print(f"[attn-stats] {args.dataset}/{args.proto} sampling {len(sample_idx)}/{N}")

    model, _ = clip.load(args.backbone, device="cuda", jit=False)
    model.eval()
    patch_attention(model)
    L = model.visual.positional_embedding.shape[0]
    grid = int(round((L - 1) ** 0.5))
    print(f"  L={L} grid={grid}")

    records = []
    for cnt, idx in enumerate(sample_idx):
        img = Image.open(paths[idx]).convert("RGB")
        for s in SCALES:
            x = make_crop(img, s / 10.0, seed=2024 + int(idx) * 100 + s)[None].cuda()
            _CAPTURED.clear()
            with torch.no_grad():
                f = model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                lg = f @ protos.T.cuda()
                pred = int(lg.argmax(-1).item())
            attn = rollout_attn(list(_CAPTURED), L).numpy()   # (L-1,)
            p = attn / max(attn.sum(), 1e-12)
            H = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(len(p)))
            top5 = float(np.sort(p)[-5:].sum())
            yy, xx = np.mgrid[0:grid, 0:grid]
            cy = float((p.reshape(grid, grid) * yy).sum())
            cx = float((p.reshape(grid, grid) * xx).sum())
            c = (grid - 1) / 2.0
            cent_dist = float(np.sqrt((cy - c) ** 2 + (cx - c) ** 2) / (c * np.sqrt(2)))
            records.append({"idx": int(idx), "scale": s, "pred": pred,
                            "correct": bool(pred == labels[idx]),
                            "H": H, "top5": top5, "cent_dist": cent_dist})
        if (cnt + 1) % 100 == 0:
            print(f"  {cnt+1}/{len(sample_idx)}")

    # ---- 汇聚 ----
    summary = {"dataset": args.dataset, "proto": args.proto,
               "n_samples": int(len(sample_idx)), "records": records}
    by_scale = {}
    for s in SCALES:
        rs = [r for r in records if r["scale"] == s]
        ok = [r for r in rs if r["correct"]]
        bad = [r for r in rs if not r["correct"]]
        by_scale[s] = {
            "H_all": float(np.mean([r["H"] for r in rs])),
            "H_correct": float(np.mean([r["H"] for r in ok])) if ok else None,
            "H_wrong": float(np.mean([r["H"] for r in bad])) if bad else None,
            "n_correct": len(ok), "n_wrong": len(bad),
            "top5_all": float(np.mean([r["top5"] for r in rs])),
            "cent_dist_all": float(np.mean([r["cent_dist"] for r in rs])),
        }
    summary["by_scale"] = by_scale
    out_dir = ROOT / "outputs" / "attn_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}.json"
    with open(p, "w") as f:
        json.dump(summary, f)
    print(f"[saved] {p}")
    for s in SCALES:
        b = by_scale[s]
        print(f"  s={s:2d} H_correct={b['H_correct']:.3f} "
              f"H_wrong={b['H_wrong']:.3f} (n={b['n_correct']}/{b['n_wrong']})")


if __name__ == "__main__":
    main()
