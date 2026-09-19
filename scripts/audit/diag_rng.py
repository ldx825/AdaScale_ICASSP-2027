"""
diag_rng.py
-----------
隔离诊断：官方脚本流程（seed→load→text_emb→逐张 crop+forward）的前 N 张
scale-1 特征是否与官方缓存一致。

如果一致 ⇒ 官方 RNG 流复现正确，batch 版差异来自其实现细节（DataLoader 等）。
如果不一致 ⇒ 存在隐藏的 RNG 消耗差异。

用法:
  python scripts/audit/diag_rng.py --dataset PET --backbone ViT-B/32 --n 20 --scale 1
"""
import os
import sys
import random
import argparse

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LGCLIP_DIR = os.path.join(PROJECT_ROOT, "third_party", "LG-CLIP")
sys.path.insert(0, LGCLIP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import clip  # noqa: E402
from utils.myDataset import PETDataset, EUROSATDataset, FLODataset  # noqa: E402

DATASET_MAP = {"PET": PETDataset, "EUROSAT": EUROSATDataset, "FLO": FLODataset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--n", default=20, type=int)
    ap.add_argument("--scale", default=1, type=int)
    ap.add_argument("--seed", default=2024, type=int)
    ap.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_text", action="store_true",
                    help="跳过 text embedding 步骤（测试其 RNG 消耗影响）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    clip_model, _ = clip.load(args.backbone, device=args.device)
    clip_model.eval()

    ds_args = argparse.Namespace(image_root=args.image_root)
    mydataset = DATASET_MAP[args.dataset](ds_args)
    all_names = mydataset.all_names
    testdf = list(zip(list(mydataset.test_files)[:args.n],
                      mydataset.test_labels.numpy()[:args.n]))

    if not args.skip_text:
        texts = [f"A photo of a {c}." for c in all_names]
        tokens = clip.tokenize(texts, context_length=77).to(args.device)
        with torch.no_grad():
            _ = clip_model.encode_text(tokens).float()

    sf = args.scale / 10.0
    tf = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(sf, sf),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    feats = []
    for img_path, label in testdf:
        img = Image.open(img_path).convert('RGB')
        img = tf(img)
        with torch.no_grad():
            f = clip_model.encode_image(img.unsqueeze(0).to(args.device)).float()
            f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    feats = torch.cat(feats).numpy()

    bk = args.backbone.replace("-", "").replace("/", "")
    off_path = os.path.join(args.image_root, args.dataset, "multi_scale",
                            f"CLIP_{bk}_feature_scale{args.scale}.hdf5")
    with h5py.File(off_path, "r") as f:
        off = np.array(f["test_f"])[:args.n]

    diff = np.abs(off - feats)
    print(f"n={args.n} scale={args.scale} skip_text={args.skip_text}")
    print(f"  max|diff| = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}")
    row_max = diff.max(axis=1)
    n_bad = (row_max > 1e-4).sum()
    print(f"  rows > 1e-4: {n_bad}/{args.n}")
    if n_bad:
        bad = np.where(row_max > 1e-4)[0][:10]
        print(f"  first bad rows: {bad.tolist()}")


if __name__ == "__main__":
    main()
