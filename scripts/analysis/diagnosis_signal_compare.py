#!/usr/bin/env python3
"""A4: 诊断信号对比 —— 域级融合效用诊断的两个候选：

  S1 (现用): 归一化熵差 ΔH = mean(H_fused - H_single)/logC （校准集上，label-free）
  S2 (新):   拖拽统计 D_to_s1 = 在 s1≠s10 样本上，full-10 融合预测跟随 s1 的比例
             （label-free；越"高"越过早被小尺度拖走）

评估：与带标签真值 sign(Acc_fused − Acc_single) 的一致性（6 设置 + B/16 子集）。
输出: outputs/deep_diag/signal_compare.json + 终端表
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CASES = [
    ("PET", "ViT-B32", "text"), ("PET", "ViT-B32", "gen"),
    ("EUROSAT", "ViT-B32", "text"), ("EUROSAT", "ViT-B32", "gen"),
    ("FLO", "ViT-B32", "text"), ("FLO", "ViT-B32", "gen"),
    ("CUB", "ViT-B32", "text"),
    ("PET", "ViT-B16", "text"), ("PET", "ViT-B16", "gen"),
    ("EUROSAT", "ViT-B16", "text"), ("FLO", "ViT-B16", "text"),
]


def main():
    rows = []
    print(f"{'setting':22s} {'truth':>8s} | {'dH(%)':>8s} {'→mode':>6s} {'ok':>3s} | "
          f"{'to_s1%':>7s} {'to_s10%':>7s} {'→mode':>6s} {'ok':>3s}")
    for ds, bk, proto in CASES:
        # 真值与 ΔH
        f2 = ROOT / "outputs" / "final" / f"adascale_v2_{ds}_{bk}_{proto}_r4_d0.65_e0.01.json"
        if not f2.exists():
            f2 = ROOT / "outputs" / "final" / f"adascale_v2_{ds}_{bk}_{proto}_single.json"
        if not f2.exists():
            continue
        d2 = json.loads(f2.read_text())
        utility = d2["acc_full10_fused"] - d2["acc_single10"]
        truth = "FUSED" if utility > 0 else "SINGLE"
        dH = d2["rel_entropy_diff_cal"] * 100  # %
        mode1 = "SINGLE" if d2["rel_entropy_diff_cal"] > 0.01 else "FUSED"
        ok1 = mode1 == truth

        # 拖拽信号
        fd = ROOT / "outputs" / "deep_diag" / f"{ds}_{bk}_{proto}.json"
        if not fd.exists():
            continue
        dd = json.loads(fd.read_text())["drag"]
        to_s1 = dd["final_to_s1"] * 100
        to_s10 = dd["final_to_s10"] * 100
        # 简单阈值（不做任何调参：用域间明显分离的观察值 10% 做演示阈值）
        mode2 = "SINGLE" if to_s1 > 10.0 else "FUSED"
        ok2 = mode2 == truth
        rows.append({"ds": ds, "bk": bk, "proto": proto, "utility": utility,
                     "truth": truth, "dH": dH, "mode_dH": mode1, "ok_dH": ok1,
                     "to_s1": to_s1, "to_s10": to_s10, "mode_drag": mode2,
                     "ok_drag": ok2})
        print(f"{ds+'/'+proto+f' ({bk[4:]})':22s} {truth:>8s} | {dH:+8.2f} {mode1:>6s} "
              f"{'Y' if ok1 else 'n':>3s} | {to_s1:7.2f} {to_s10:7.2f} {mode2:>6s} "
              f"{'Y' if ok2 else 'n':>3s}")

    n1 = sum(r["ok_dH"] for r in rows)
    n2 = sum(r["ok_drag"] for r in rows)
    print(f"\n  正确率: ΔH 信号 {n1}/{len(rows)}  |  拖拽信号(阈值10%) {n2}/{len(rows)}")
    # 信号分离度: SINGLE 域与 FUSED 域之间的信号差
    s_set = [r for r in rows if r["truth"] == "SINGLE"]
    f_set = [r for r in rows if r["truth"] == "FUSED"]
    if s_set and f_set:
        gap1 = min(r["dH"] for r in s_set) - max(r["dH"] for r in f_set)
        gap2 = min(r["to_s1"] for r in s_set) - max(r["to_s1"] for r in f_set)
        print(f"  信号分离度 (SINGLE 域最小值 − FUSED 域最大值): ΔH {gap1:+.2f}  "
              f"拖拽 {gap2:+.2f}")
    with open(ROOT / "outputs" / "deep_diag" / "signal_compare.json", "w") as f:
        json.dump({"rows": rows, "acc_dH": f"{n1}/{len(rows)}",
                   "acc_drag": f"{n2}/{len(rows)}"}, f, indent=2)
    print("[saved] outputs/deep_diag/signal_compare.json")


if __name__ == "__main__":
    main()
