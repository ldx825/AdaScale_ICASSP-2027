#!/usr/bin/env python3
"""Paired bootstrap 统计检验（docs §19）。

对同一测试集上的任意两个 policy（Full-10 / AdaScale / Fixed-K / MarginOnly）
做 paired bootstrap：
- 采样重排样本索引（共享重排 → paired）
- 计算 Δacc = acc(A) − acc(B) 的 bootstrap 分布 → 95% CI + two-sided p

输出: outputs/stats/<ds>_<bk>_<proto>_paired.csv

用法:
    python scripts/final/paired_bootstrap.py --dataset PET --backbone ViT-B32 --proto text
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # simulate.py imports src.utils.*

from src.adascale.simulate import (  # noqa: E402
    FixedKPolicy, MarginOnlyPolicy, StabilityPolicy, simulate,
)
from src.utils.load_features import load_real_scales, load_gen_scales  # noqa: E402

B_DEFAULT = 10000
SEED = 0


def load_features(dataset: str, backbone: str, proto: str,
                  image_root: Path | None = None, ngen: int = 10):
    image_root = str(image_root or (ROOT / "data"))
    feats, labels, text_emb = load_real_scales(image_root, dataset, backbone)
    if proto == "text":
        prototypes = text_emb
    else:
        gen_feats, _ = load_gen_scales(image_root, dataset, backbone, "", ngen)
        from src.utils.load_features import full_scale_aggregate
        gf = full_scale_aggregate(gen_feats)
        prototypes = gf.mean(dim=1)
    return feats, np.asarray(labels.numpy()), prototypes


def per_sample_correct(feats, prototypes, policy, labels):
    res = simulate(feats, prototypes, policy)
    return (res.pred == labels).astype(np.float64), res.avg_scales


def paired_bootstrap(corr_a: np.ndarray, corr_b: np.ndarray,
                     b: int = B_DEFAULT, seed: int = SEED):
    """Δacc = A − B 的 paired bootstrap。返回 (mean, lo95, hi95, p_two_sided)。"""
    n = len(corr_a)
    diff = corr_a - corr_b
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    # 向量化 bootstrap：b × n 索引
    idx = rng.integers(0, n, size=(b, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p：Δ=0 在分布中的位置（两倍单边）
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = min(p, 1.0)
    return obs, float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--b", type=int, default=B_DEFAULT)
    ap.add_argument("--ngen", type=int, default=10)
    ap.add_argument("--adascale-config", default=None,
                    help="JSON like {\"r\":4,\"delta\":0.25,\"eps\":0.01}; "
                         "default: current frozen candidate")
    args = ap.parse_args()

    feats, labels, prototypes = load_features(args.dataset, args.backbone,
                                              args.proto, ngen=args.ngen)
    n = len(labels)
    print(f"[bootstrap] {args.dataset}/{args.backbone}/{args.proto}: N={n}")

    # AdaScale 配置：优先 --adascale-config，默认使用冻结候选
    if args.adascale_config:
        cfg = json.loads(args.adascale_config)
    else:
        cfg = {"r": 4, "delta": 0.25, "eps": 0.01}
    policies = {
        "Full-10": FixedKPolicy(10),
        f"AdaScale(r{cfg['r']},d{cfg['delta']},e{cfg['eps']})":
            StabilityPolicy(r=cfg["r"], delta=cfg["delta"], eps=cfg["eps"],
                            min_scale=1, use_js=True, use_margin=True),
        "Fixed-5": FixedKPolicy(5),
        "Fixed-7": FixedKPolicy(7),
        "Fixed-9": FixedKPolicy(9),
        "MarginOnly(0.3)": MarginOnlyPolicy(delta=0.3, min_scale=3),
    }

    corr: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for name, pol in policies.items():
        c, s = per_sample_correct(feats, prototypes, pol, labels)
        corr[name] = c
        scales[name] = s
        print(f"  {name:38s} acc={c.mean()*100:6.2f}%  avg_scales={s:5.2f}")

    out_dir = ROOT / "outputs" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_cfg = f"r{cfg['r']}_d{cfg['delta']}_e{cfg['eps']}"
    out_csv = out_dir / f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_{tag_cfg}_paired.csv"

    rows = []
    for a, b in itertools.combinations(corr.keys(), 2):
        obs, lo, hi, p = paired_bootstrap(corr[a], corr[b], b=args.b)
        rows.append({
            "policy_A": a, "policy_B": b,
            "acc_A": corr[a].mean(), "acc_B": corr[b].mean(),
            "delta_acc": obs, "ci95_lo": lo, "ci95_hi": hi, "p_two_sided": p,
            "avg_scales_A": scales[a], "avg_scales_B": scales[b],
            "significant_5pct": p < 0.05,
        })

    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}")

    # 关键比较打印
    print("\n[key comparisons]")
    for r in rows:
        if ("AdaScale" in r["policy_A"] and r["policy_B"] == "Full-10") or \
           ("AdaScale" in r["policy_B"] and r["policy_A"] == "Full-10"):
            print(f"  {r['policy_A']} vs {r['policy_B']}: "
                  f"Δ={r['delta_acc']*100:+.2f}% "
                  f"[{r['ci95_lo']*100:+.2f}, {r['ci95_hi']*100:+.2f}] "
                  f"p={r['p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
