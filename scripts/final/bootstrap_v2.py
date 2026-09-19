#!/usr/bin/env python3
"""AdaScale v2 的配对 bootstrap 显著性检验（6 设置）。

直接从 outputs/final/adascale_v2_*_preds.npz 加载每样本预测，
对 fixed-K 补跑 simulate，然后做关键对比的 paired bootstrap。

输出: outputs/stats/<ds>_<bk>_<proto>_v2_paired.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate,
)
from src.adascale.simulate import FixedKPolicy, simulate  # noqa: E402

B = 10000
SEED = 0


def paired_bootstrap(corr_a, corr_b, b=B, seed=SEED):
    d = corr_a.astype(np.float64) - corr_b.astype(np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(lo), float(hi), min(float(p), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text", choices=["text", "gen"])
    ap.add_argument("--stab", default="4,0.45,0.01")
    args = ap.parse_args()

    base = f"adascale_v2_{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}"
    r_s, d_s, e_s = [float(x) for x in args.stab.split(",")]
    stab_tag = f"r{int(r_s)}_d{d_s}_e{e_s}"
    out_dir = ROOT / "outputs" / "final"
    candidates = [out_dir / f"{base}_{stab_tag}_preds.npz",
                  out_dir / f"{base}_single_preds.npz"]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(f"none of {candidates} exist")
    npz = np.load(npz_path)
    print(f"[bootstrap-v2] loading {npz_path.name}")
    labels = npz["labels"]
    preds = {
        "AdaScale-v2": npz["pred_v2"],
        "Full-10(fused)": npz["pred_fused"],
        "Single-10": npz["pred_single"],
    }
    # AdaScale-v3（若存在）
    v3_path = out_dir / f"adascale_v3_{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_{stab_tag}_preds.npz"
    if v3_path.exists():
        npz3 = np.load(v3_path)
        assert np.array_equal(labels, npz3["labels"])
        preds["AdaScale-v3"] = npz3["pred_v3"]
        print(f"  + AdaScale-v3 from {v3_path.name}")

    # fixed-K per-sample
    root = str(ROOT / "data")
    feats, lab2, text_emb = load_real_scales(root, args.dataset, args.backbone)
    if args.proto == "text":
        prototypes = text_emb
    else:
        gf, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        prototypes = full_scale_aggregate(gf).mean(dim=1)
    assert np.array_equal(labels, lab2.numpy())
    for k in (5, 7):
        r = simulate(feats, prototypes, FixedKPolicy(k))
        preds[f"Fixed-{k}"] = r.pred

    corr = {k: (v == labels).astype(np.float64) for k, v in preds.items()}
    for k, v in corr.items():
        print(f"  {k:16s} acc={v.mean()*100:6.2f}%")

    rows = []
    for a, b in itertools.combinations(corr.keys(), 2):
        obs, lo, hi, p = paired_bootstrap(corr[a], corr[b])
        rows.append(dict(policy_A=a, policy_B=b, acc_A=corr[a].mean(),
                         acc_B=corr[b].mean(), delta=obs, ci_lo=lo, ci_hi=hi,
                         p=p, sig=(p < 0.05)))
    import csv
    tag = f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}"
    out = ROOT / "outputs" / "stats" / f"{tag}_v2_{stab_tag}_paired.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")
    for r in rows:
        if r["policy_A"] in ("AdaScale-v2", "AdaScale-v3") or \
           r["policy_B"] in ("AdaScale-v2", "AdaScale-v3"):
            print(f"  {r['policy_A']} vs {r['policy_B']}: Δ={r['delta']*100:+.2f}% "
                  f"[{r['ci_lo']*100:+.2f},{r['ci_hi']*100:+.2f}] p={r['p']:.4f}"
                  f"{' *' if r['sig'] else ''}")


if __name__ == "__main__":
    main()
