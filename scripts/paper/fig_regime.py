#!/usr/bin/env python3
"""论文 Figure 2：两 regime 前缀精度曲线（真矢量，TrueType/Type 42 嵌入）。

左：Oxford-Pets/text —— 融合(prefix) 高于单尺度 → FUSED 模式；
右：EuroSAT/text   —— 融合低于单尺度     → SINGLE 模式。

设计尺寸≈最终显示尺寸（约 0.47 栏宽），字号按显示大小设置，
以 TrueType (Type 42) 嵌入，避免旧版 Type 3（位图化字形）发虚问题。

数据:  outputs/scale_audit/{PET,EUROSAT}/ViT-B32/summary_text.json
输出:  figures/figA_{pet,eurosat}_text.{pdf,png} + paper/figures/ 副本
用法:  python scripts/paper/fig_regime.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType 矢量字体
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "outputs" / "scale_audit"
OUT = ROOT / "figures"
PAPER = ROOT / "paper" / "figures"

BLUE = "#2166ac"
ORANGE = "#d95f02"


def panel(dataset: str, title: str, fname: str):
    sm = json.loads((AUD / dataset / "ViT-B32" / "summary_text.json").read_text())
    pref = np.array(sm["prefix_acc_curve"]) * 100
    single = np.array(sm["per_scale_independent_acc"]) * 100
    ts = np.arange(1, len(pref) + 1)

    fig, ax = plt.subplots(figsize=(1.62, 1.14))
    ax.plot(ts, pref, "-o", color=BLUE, lw=1.2, ms=2.6, label="Prefix (fusion)")
    ax.plot(ts, single, "--s", color=ORANGE, lw=1.0, ms=2.4, label="Single scale")
    ax.set_xlabel("# scales evaluated", fontsize=5.6)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=5.6)
    ax.set_title(title, fontsize=5.8)
    ax.set_xticks(range(1, 11))
    ax.tick_params(labelsize=4.9, pad=1.0)
    ax.grid(alpha=0.25, lw=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=4.6, loc="upper left", framealpha=0.92,
              handlelength=1.2, handletextpad=0.4, borderpad=0.25)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    fig.tight_layout(pad=0.2)
    for d in (OUT, PAPER):
        d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=300)
    shutil.copy(OUT / f"{fname}.pdf", PAPER / f"{fname}.pdf")
    plt.close(fig)
    print(f"[saved] {fname}  fusion(end)={pref[-1]:.2f}  "
          f"single(end)={single[-1]:.2f}")


def main():
    panel("PET", "Oxford-Pets (text)", "figA_pet_text")
    panel("EUROSAT", "EuroSAT (text)", "figA_eurosat_text")


if __name__ == "__main__":
    main()
