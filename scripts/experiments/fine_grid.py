#!/usr/bin/env python3
"""Fine stability grid（δ/ε 灵敏度）生成，供 plot_sensitivity 使用。

网格：r∈{3,4} × δ∈{0.25,0.35,0.45,0.55,0.65} × ε∈{1.0(off),0.03,0.01}
输出: outputs/methods/<ds>/<bk>/fine_grid_<proto>.csv
（含 fixed_ref 列的协议与 PET/gen 现有格式一致）

用法: python scripts/experiments/fine_grid.py --dataset FLO --proto text
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate,
)
from src.adascale.simulate import (  # noqa: E402
    StabilityPolicy, FixedKPolicy, simulate_and_eval,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    args = ap.parse_args()

    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gf, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gf).mean(dim=1)
    labels = labels.numpy()

    # fixed-K 参考曲线（用于"同预算 gain"）
    fixed_rows = {}
    for k in range(1, 11):
        r = simulate_and_eval(feats, prototypes, FixedKPolicy(k), labels)
        fixed_rows[k] = r.acc

    rows = []
    for r_, delta, eps in itertools.product(
            [3, 4], [0.25, 0.35, 0.45, 0.55, 0.65], [1.0, 0.03, 0.01]):
        pol = StabilityPolicy(r=r_, delta=delta, eps=eps, min_scale=1,
                              use_js=(eps < 1.0), use_margin=True)
        res = simulate_and_eval(feats, prototypes, pol, labels)
        # 同预算 fixed 插值
        k_lo = int(res.avg_scales)
        frac = res.avg_scales - k_lo
        ref = fixed_rows[k_lo] + frac * (fixed_rows[min(k_lo + 1, 10)] - fixed_rows[k_lo])
        rows.append(dict(method=f"r{r_}_d{delta}_e{eps}", acc=res.acc,
                         avg_scales=res.avg_scales, fixed_ref_acc=ref,
                         gain=res.acc - ref))
        print(f"  r{r_}_d{delta}_e{eps}: acc={res.acc*100:.2f}% "
              f"scales={res.avg_scales:.2f} gain={ (res.acc-ref)*100:+.3f}")

    out_dir = ROOT / "outputs" / "methods" / args.dataset / \
        args.backbone.replace("/", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fine_grid_{args.proto}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("[saved]", out)


if __name__ == "__main__":
    main()
