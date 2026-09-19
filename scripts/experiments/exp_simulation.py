"""
exp_simulation.py
-----------------
E04/E05：在缓存的 per-scale 特征上评估：
  - Fixed-K baseline（K=1..10）
  - 稳定性/置信度自适应早停策略网格
  - 输掉 Acc–compute 曲线数据（JSON + CSV）

policy 定义（见 src/adascale/simulate.py）：
  StabilityPolicy(r, delta, eps): 连续 r 个 scale top-1 一致 & margin>=delta & JS<=eps

用法:
  python scripts/experiments/exp_simulation.py --dataset PET --backbone ViT-B/32 --proto text
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.load_features import (
    load_real_scales, load_gen_scales, full_scale_aggregate,
)
from src.adascale.simulate import (
    FixedKPolicy, StabilityPolicy, MarginOnlyPolicy, simulate_and_eval,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PET")
    p.add_argument("--backbone", default="ViT-B/32")
    p.add_argument("--proto", default="text", choices=["text", "gen"])
    p.add_argument("--llm", default="")
    p.add_argument("--ngen", default=10, type=int)
    p.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "outputs", "methods"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_all(args):
    feats, labels, text_emb = load_real_scales(args.image_root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
        note = "text"
    else:
        gen_feats, _ = load_gen_scales(args.image_root, args.dataset, args.backbone,
                                       args.llm, args.ngen)
        gf = full_scale_aggregate([g.to(args.device) for g in gen_feats])
        prototypes = gf.mean(dim=1)
        note = f"gen(llm={args.llm!r},n={args.ngen})"
    return feats, labels.numpy(), prototypes, note


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, args.dataset, args.backbone.replace("/", "")),
                exist_ok=True)
    out_base = os.path.join(args.out_dir, args.dataset, args.backbone.replace("/", ""))
    feats, labels, prototypes, note = load_all(args)
    print(f"[sim] {args.dataset}/{args.backbone}/{args.proto} N={len(labels)}")

    rows = []

    # ---- Fixed-K ----
    for k in range(1, 11):
        res = simulate_and_eval(feats, prototypes, FixedKPolicy(k), labels,
                                num_scales=10, device=args.device)
        rows.append(dict(method=f"fixed-{k}", kind="fixed", params=json.dumps({"k": k}),
                         acc=res.acc, avg_scales=res.avg_scales))
        print(f"  fixed-{k}: acc={res.acc*100:.2f}% avg_scales={res.avg_scales:.2f}")

    # ---- Stability 网格 ----
    grid = list(itertools.product(
        [2, 3],                # r
        [0.0, 0.2, 0.35, 0.5, 0.65],  # margin delta
        [1.0, 0.1, 0.05, 0.02],       # js eps (1.0 ≈ 关闭)
        [1, 2],               # min_scale
    ))
    for r, delta, eps, min_s in grid:
        pol = StabilityPolicy(r=r, delta=delta, eps=eps, min_scale=min_s,
                              use_js=(eps < 1.0), use_margin=(delta > 0))
        res = simulate_and_eval(feats, prototypes, pol, labels,
                                num_scales=10, device=args.device)
        tag = f"stab_r{r}_d{delta}_e{eps}_m{min_s}"
        rows.append(dict(method=tag, kind="stability",
                         params=json.dumps({"r": r, "delta": delta, "eps": eps, "min_scale": min_s}),
                         acc=res.acc, avg_scales=res.avg_scales))
        print(f"  {tag}: acc={res.acc*100:.2f}% avg_scales={res.avg_scales:.2f}")

    # ---- Margin-only（消融） ----
    for delta in [0.2, 0.35, 0.5, 0.65]:
        res = simulate_and_eval(feats, prototypes, MarginOnlyPolicy(delta=delta, min_scale=2),
                                labels, num_scales=10, device=args.device)
        rows.append(dict(method=f"margin_d{delta}", kind="margin_only",
                         params=json.dumps({"delta": delta}), acc=res.acc,
                         avg_scales=res.avg_scales))
        print(f"  margin_d{delta}: acc={res.acc*100:.2f}% avg_scales={res.avg_scales:.2f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_base, f"sim_results_{args.proto}.csv")
    df.to_csv(out_csv, index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
