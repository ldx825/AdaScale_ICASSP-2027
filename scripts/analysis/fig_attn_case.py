#!/usr/bin/env python3
"""Paper-ready 单案例 attention 图：1 行 × 4 尺度 × [crop | rollout overlay]。

用法:
  python scripts/analysis/fig_attn_case.py --dataset EUROSAT --proto text \
      --case_idx 15057 --out fig_attn_eurosat
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from attention_maps import (  # noqa: E402
    patch_attention, rollout_attn, make_crop, denorm, _CAPTURED,
)
from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate,
)

LOGIT_TAU = 0.01
SCALES_SHOW = [1, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="EUROSAT")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text")
    ap.add_argument("--case_idx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale_set", default="1,3,5,10")
    args = ap.parse_args()
    global SCALES_SHOW
    SCALES_SHOW = [int(x) for x in args.scale_set.split(",")]

    root = str(ROOT / "data")
    feats, labels, text_emb = load_real_scales(root, args.dataset, args.backbone)
    labels = labels.numpy()
    protos = text_emb if args.proto == "text" else \
        full_scale_aggregate(load_gen_scales(root, args.dataset, args.backbone,
                                             "", 10)[0]).mean(dim=1)

    if args.dataset == "EUROSAT":
        split = json.load(open("data/EUROSAT/split_EuroSAT.json"))
        paths = [str(Path("data/EUROSAT/2750") / e[0]) for e in split["test"]]
    elif args.dataset == "PET":
        split = json.load(open("data/PET/OxfordPets/split_OxfordPets.json"))
        paths = [str(Path("data/PET/OxfordPets/images") / e[0]) for e in split["test"]]
    names = [e[2] for e in split["test"]]
    label2name = {e[1]: e[2] for e in split["test"]}

    model, _ = clip.load(args.backbone, device="cuda", jit=False)
    model.eval()
    patch_attention(model)
    L = model.visual.positional_embedding.shape[0]
    grid = int(round((L - 1) ** 0.5))

    idx = args.case_idx
    img = Image.open(paths[idx]).convert("RGB")
    fig, axes = plt.subplots(1, len(SCALES_SHOW) * 2,
                             figsize=(0.80 * len(SCALES_SHOW) * 2, 0.98))
    for c, s in enumerate(SCALES_SHOW):
        x = make_crop(img, s / 10.0, seed=2024 + idx * 100 + s)[None].cuda()
        _CAPTURED.clear()
        with torch.no_grad():
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            lg = f @ protos.T.cuda()
            prob = torch.softmax(lg / LOGIT_TAU, dim=-1)
            pred = int(lg.argmax(-1))
            conf = float(prob[0, pred])
        attn = rollout_attn(list(_CAPTURED), L).numpy()
        amap = attn.reshape(grid, grid)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-9)
        vis = denorm(x.cpu())[0].permute(1, 2, 0).numpy()
        ok = "OK" if pred == labels[idx] else "WRONG"
        ax = axes[c * 2]
        ax.imshow(vis); ax.axis("off")
        ax.set_title(f"s={s}\nGT={names[idx]}", fontsize=5.4)
        ax2 = axes[c * 2 + 1]
        ax2.imshow(vis)
        ax2.imshow(amap, cmap="jet", alpha=0.45, extent=(0, 224, 224, 0),
                   interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title(f"→{label2name[pred]} {conf:.2f}\n[{ok}]", fontsize=5.4)
    fig.tight_layout(pad=0.12)
    out = ROOT / "figures" / "deep_diag" / args.out
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
