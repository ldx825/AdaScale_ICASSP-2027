#!/usr/bin/env python3
"""收集 AdaScale v2 最终结果 → outputs/final/all_results_v2.csv

读取 outputs/final/adascale_v2_*.json（最终配置 r4,d0.45,e0.01），
合并 bootstrap 显著性，输出论文主表所需的统一 CSV。

用法:
    python scripts/final/collect_results.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAB = "r4_d0.65_e0.01"
DATASETS = ["PET", "EUROSAT", "FLO", "CUB"]
PROTOS = ["text", "gen"]


def main():
    rows = []
    for ds in DATASETS:
        for proto in PROTOS:
            base = f"adascale_v2_{ds}_ViT-B32_{proto}"
            p = ROOT / "outputs" / "final" / f"{base}_{STAB}.json"
            if not p.exists():
                p = ROOT / "outputs" / "final" / f"{base}_single.json"
            if not p.exists():
                print(f"[warn] missing {base}")
                continue
            d = json.loads(p.read_text())
            v2 = d["adascale_v2"]

            # bootstrap 显著性（AdaScale vs Full-10）
            bp = ROOT / "outputs" / "stats" / f"{ds}_ViT-B32_{proto}_v2_{STAB}_paired.csv"
            delta_p = None
            v3_p = None
            if bp.exists():
                with open(bp) as f:
                    for r in csv.DictReader(f):
                        if r["policy_A"] == "AdaScale-v2" and r["policy_B"] == "Full-10(fused)":
                            delta_p = float(r["p"])
                        if {r["policy_A"], r["policy_B"]} == {"AdaScale-v3", "Full-10(fused)"}:
                            v3_p = float(r["p"])
            # v3 per-sample（若存在；SINGLE 模式无 v3 分文件 → 等同 v2）
            v3_acc = v2["acc"]
            v3_npz = ROOT / "outputs" / "final" / \
                f"adascale_v3_{ds}_ViT-B32_{proto}_{STAB}_preds.npz"
            if v3_npz.exists():
                z3 = np.load(v3_npz)
                v3_acc = float((z3["pred_v3"] == z3["labels"]).mean())
            rows.append({
                "dataset": ds, "proto": proto, "mode": d["mode"],
                "baseline_acc": d["acc_full10_fused"],
                "baseline_scales": 10.0,
                "single_acc": d["acc_single10"],
                "adascale_acc": v2["acc"],
                "adascale_acc_excl_cal": v2["acc_excl_cal"],
                "adascale_scales": v2["avg_scales"],
                "v3_acc": v3_acc,
                "v3_delta_vs_baseline": v3_acc - d["acc_full10_fused"],
                "v3_p_vs_baseline": v3_p if v3_p is not None else delta_p,
                "scale_saving": 1 - v2["avg_scales"] / 10.0,
                "delta_vs_baseline": v2["acc"] - d["acc_full10_fused"],
                "p_vs_baseline": delta_p,
                "rel_entropy_diff_cal": d["rel_entropy_diff_cal"],
                "fixed5_acc": d["fixed"].get("fixed-5", {}).get("acc"),
                "fixed7_acc": d["fixed"].get("fixed-7", {}).get("acc"),
                "fixed9_acc": d["fixed"].get("fixed-9", {}).get("acc"),
            })

    out = ROOT / "outputs" / "final" / "all_results_v2.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[saved] {out} ({len(rows)} rows)")
    for r in rows:
        ps = f"p={r['p_vs_baseline']:.4f}" if r["p_vs_baseline"] is not None else ""
        print(f"  {r['dataset']:8s}/{r['proto']:4s} {r['mode']:6s} "
              f"base={r['baseline_acc']*100:6.2f} v2={r['adascale_acc']*100:6.2f} "
              f"({r['delta_vs_baseline']*100:+5.2f}, {ps}) sc={r['adascale_scales']:.2f}")


if __name__ == "__main__":
    main()
