"""
diag_rng2.py
------------
进程内 A/B 实验：直接循环 vs DataLoader 循环 的 crop 序列是否一致。

用法: python scripts/audit/diag_rng2.py --dataset PET --n 64
"""
import os
import sys
import random
import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LGCLIP_DIR = os.path.join(PROJECT_ROOT, "third_party", "LG-CLIP")
sys.path.insert(0, LGCLIP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.myDataset import PETDataset, EUROSATDataset, FLODataset  # noqa: E402

DATASET_MAP = {"PET": PETDataset, "EUROSAT": EUROSATDataset, "FLO": FLODataset}


def make_transform(sf):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(sf, sf),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])


class _DS(Dataset):
    def __init__(self, pairs, tf):
        self.pairs = pairs
        self.tf = tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p, lab = self.pairs[i]
        img = Image.open(p).convert('RGB')
        img = self.tf(img)
        return img, lab, i


def run_direct(pairs, sf, batch=8):
    tf = make_transform(sf)
    feats = []
    for p, lab in pairs:
        img = Image.open(p).convert('RGB')
        img = tf(img)
        feats.append(img)
    return torch.stack(feats)


def run_dataloader(pairs, sf, batch=8, workers=0):
    tf = make_transform(sf)
    ds = _DS(pairs, tf)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=lambda b: b)
    feats, order = [], []
    for batch_ in dl:
        imgs = torch.stack([b[0] for b in batch_])
        feats.append(imgs)
        order.extend([b[2] for b in batch_])
    feats = torch.cat(feats)
    if order != list(range(len(order))):
        inv = np.argsort(order)
        feats = feats[inv]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--n", default=64, type=int)
    ap.add_argument("--scale", default=1, type=int)
    ap.add_argument("--seed", default=2024, type=int)
    ap.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    args = ap.parse_args()

    ds_args = argparse.Namespace(image_root=args.image_root)
    mydataset = DATASET_MAP[args.dataset](ds_args)
    pairs = list(zip(list(mydataset.test_files)[:args.n],
                     mydataset.test_labels.numpy()[:args.n]))
    sf = args.scale / 10.0

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    a = run_direct(pairs, sf)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    b = run_dataloader(pairs, sf, batch=16, workers=0)

    diff = (a - b).abs()
    print(f"direct-vs-dataloader  n={args.n} scale={args.scale}")
    print(f"  max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}")
    print(f"  identical: {torch.allclose(a, b, atol=1e-6)}")

    # 每样本整图差异，找出首个不一致的样本
    per_sample = diff.flatten(1).max(1).values.numpy()
    bad = np.where(per_sample > 1e-6)[0]
    print(f"  bad samples: {bad[:10].tolist()} (total {len(bad)})")


if __name__ == "__main__":
    main()
