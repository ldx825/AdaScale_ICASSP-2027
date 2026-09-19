#!/usr/bin/env python3
"""Attention 可视化：跨尺度 crop 的 CLIP 注意力 rollout + 预测。

产出论文用案例图：
  行 = 案例（EUROSAT 被晚期尺度伤害的样本 / PET 被融合救回的样本）
  列 = scale ∈ {1,3,5,10}：crop 原图 | attention rollout overlay + 预测

用法（GPU）:
  python scripts/analysis/attention_maps.py --dataset EUROSAT --proto text --n_cases 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "LG-CLIP"))

import clip  # noqa: E402

from src.utils.load_features import (  # noqa: E402
    load_real_scales, load_gen_scales, full_scale_aggregate, SCALE_WEIGHTS,
)

LOGIT_TAU = 0.01
SCALES_SHOW = [1, 3, 5, 10]


# ---------- attention 捕获 ----------
_CAPTURED = []


def patch_attention(model):
    from clip.model import ResidualAttentionBlock

    def attention_with_weights(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        out, w = self.attn(x, x, x, need_weights=True,
                           attn_mask=self.attn_mask,
                           average_attn_weights=False)
        _CAPTURED.append(w.detach())  # (bs=1, heads, L, L)
        return out

    for blk in model.visual.transformer.resblocks:
        blk.attention = attention_with_weights.__get__(blk)
    return model


def rollout_attn(attns, L):
    """attention rollout，返回 CLS->patches (L-1,)。"""
    res = torch.eye(L, device=attns[0].device)
    for a in attns:                      # a: (1, heads, L, L)
        a = a[0].mean(0)                 # (L, L) heads 平均
        a = a + torch.eye(L, device=a.device)
        a = a / a.sum(-1, keepdim=True)
        res = a @ res
    return res[0, 1:].cpu()              # CLS row 的 patch 列


def set_seed(s):
    import random
    s = int(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_crop(img, sf, seed, size=224):
    """与提取管线一致的 RandomResizedCrop。"""
    from torchvision.transforms import RandomResizedCrop, Compose, ToTensor, Normalize
    set_seed(seed)
    tf = Compose([RandomResizedCrop(size, scale=(sf, sf), interpolation=3),
                  ToTensor(),
                  Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))])
    return tf(img)


def denorm(t):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (t * std + mean).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="EUROSAT")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--proto", default="text")
    ap.add_argument("--n_cases", type=int, default=3)
    ap.add_argument("--out_dir", default=str(ROOT / "figures" / "attention"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 特征与轨迹（找案例）----
    root = str(ROOT / "data")
    feats, labels, te = load_real_scales(root, args.dataset, args.backbone)
    if args.proto == "text":
        protos = te
    else:
        gf, _ = load_gen_scales(root, args.dataset, args.backbone, "", 10)
        protos = full_scale_aggregate(gf).mean(dim=1)
    labels_np = labels.numpy()
    T = len(feats)
    N = len(labels_np)

    ps = None
    prefix_correct = np.zeros((N, T), dtype=bool)
    prefix_margin = np.zeros((N, T), dtype=np.float32)
    for t in range(1, T + 1):
        f = feats[t - 1]
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ps = float(SCALE_WEIGHTS[t - 1]) * f if ps is None else ps + float(SCALE_WEIGHTS[t - 1]) * f
        fn = ps / ps.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lg = fn @ protos.T
        prob = F.softmax(lg / LOGIT_TAU, dim=-1)
        t2 = torch.topk(prob, 2, dim=-1).values
        prefix_correct[:, t - 1] = (lg.argmax(-1).numpy() == labels_np)
        prefix_margin[:, t - 1] = (t2[:, 0] - t2[:, 1]).numpy()

    # ---- 案例选择 ----
    harmful = np.where(prefix_correct[:, :-1].any(1) & (~prefix_correct[:, -1]))[0]
    rescued = np.where((~prefix_correct[:, 0]) & prefix_correct[:, -1])[0]
    print(f"  harmful={len(harmful)} rescued={len(rescued)}")
    if args.dataset == "EUROSAT":
        # 目标故事：小尺度单预测错（且自信）→ 大尺度单预测对
        f1 = feats[0] / feats[0].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        f10 = feats[-1] / feats[-1].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        p1 = F.softmax((f1 @ protos.T) / LOGIT_TAU, dim=-1)
        p10 = F.softmax((f10 @ protos.T) / LOGIT_TAU, dim=-1)
        c1 = (p1.argmax(-1).numpy() == labels_np)
        c10 = (p10.argmax(-1).numpy() == labels_np)
        conf1 = p1.max(-1).values.numpy()
        pool = np.where((~c1) & (c10))[0]
        pool = pool[np.argsort(-conf1[pool])]  # 小尺度自信地错
        cases = pool[:40]
        case_type = "small-scale-confidently-wrong"
    else:
        pool = rescued[np.argsort(-prefix_margin[rescued, -1])]
        cases = pool[:40]
        case_type = "rescued-by-fusion"
    print(f"  selected {case_type}: {cases.tolist()}")

    # ---- 图像路径 ----
    if args.dataset == "EUROSAT":
        split = json.load(open("data/EUROSAT/split_EuroSAT.json"))
        paths = [str(Path("data/EUROSAT/2750") / e[0]) for e in split["test"]]
        names = [e[2] for e in split["test"]]
    elif args.dataset == "PET":
        split = json.load(open("data/PET/OxfordPets/split_OxfordPets.json"))
        paths = [str(Path("data/PET/OxfordPets/images") / e[0]) for e in split["test"]]
        names = [e[2] for e in split["test"]]
    else:
        raise ValueError("add dataset")

    # label -> 类名映射（供预测标签显示）
    label2name = {}
    for e in split["test"]:
        label2name[e[1]] = e[2]

    # ---- CLIP 模型 + attention patch ----
    model, _ = clip.load(args.backbone, device="cuda", jit=False)
    model.eval()
    patch_attention(model)
    L = model.visual.positional_embedding.shape[0]  # 50 for B/32@224
    grid = int(round((L - 1) ** 0.5))
    print(f"  L={L} grid={grid}")

    # ---- 重采样一致性验证：图上 crop 的预测需与选择条件一致 ----
    def predict_crop(img, s, seed):
        x = make_crop(img, s / 10.0, seed=seed)[None].cuda()
        with torch.no_grad():
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            lg = f @ protos.T.cuda()
            prob = torch.softmax(lg / LOGIT_TAU, dim=-1)
        return int(lg.argmax(-1).item()), float(prob.max().item())

    verified = []
    n_checked = 0
    for idx in cases:
        if len(verified) >= args.n_cases:
            break
        n_checked += 1
        if n_checked > 40:
            break
        img = Image.open(paths[idx]).convert("RGB")
        seed_base = 2024 + int(idx) * 100
        p1, _ = predict_crop(img, 1, seed_base + 1)
        p10, _ = predict_crop(img, 10, seed_base + 10)
        ok_cond = (p1 != labels_np[idx]) and (p10 == labels_np[idx]) \
            if args.dataset == "EUROSAT" else (p1 != labels_np[idx] and p10 == labels_np[idx])
        if ok_cond:
            verified.append(idx)
    print(f"  verified cases (re-sampled crops consistent): {verified}")
    cases = np.array(verified) if verified else cases

    # ---- 生成图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = len(cases)
    fig, axes = plt.subplots(n_show, len(SCALES_SHOW) * 2,
                             figsize=(2.0 * len(SCALES_SHOW) * 2, 2.2 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for r, idx in enumerate(cases):
        img = Image.open(paths[idx]).convert("RGB")
        for c, s in enumerate(SCALES_SHOW):
            # 用与提取相同 seed 序列（每个样本固定 seed=idx，scale=s）
            x = make_crop(img, s / 10.0, seed=2024 + idx * 100 + s)[None].cuda()
            _CAPTURED.clear()
            with torch.no_grad():
                f = model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                lg = (f @ protos.T.cuda())
                prob = torch.softmax(lg / LOGIT_TAU, dim=-1)
                pred = int(lg.argmax(-1).item())
                conf = float(prob[0, pred].item())
            # rollout
            attn = rollout_attn(list(_CAPTURED), L)          # (L-1,)
            amap = attn.reshape(grid, grid).numpy()
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-9)

            vis = denorm(x.cpu())[0].permute(1, 2, 0).numpy()
            ax = axes[r, c * 2]
            ax.imshow(vis); ax.axis("off")
            ax.set_title(f"s={s} | GT={names[idx]}", fontsize=9)
            ax2 = axes[r, c * 2 + 1]
            ax2.imshow(vis)
            ax2.imshow(amap, cmap="jet", alpha=0.5,
                       extent=(0, 224, 224, 0), interpolation="bilinear")
            ax2.axis("off")
            correct = "OK" if pred == labels_np[idx] else "WRONG"
            ax2.set_title(f"pred={label2name[pred]} p={conf:.2f} [{correct}]", fontsize=8)
    fig.suptitle(f"{args.dataset}/{args.proto} - {case_type}", fontsize=11)
    fig.tight_layout()
    out = Path(args.out_dir) / f"{args.dataset}_{args.backbone.replace('/', '')}_{args.proto}_cases_grid"
    fig.savefig(f"{out}.png", dpi=150)
    fig.savefig(f"{out}.pdf")
    print("[saved]", out)
    # 记录案例
    meta = {"dataset": args.dataset, "proto": args.proto, "case_type": case_type,
            "cases": [{"idx": int(i), "class": names[i],
                       "prefix_correct": prefix_correct[i].tolist(),
                       "full_correct": bool(prefix_correct[i, -1])} for i in cases]}
    with open(Path(args.out_dir) / f"{args.dataset}_{args.proto}_cases_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
