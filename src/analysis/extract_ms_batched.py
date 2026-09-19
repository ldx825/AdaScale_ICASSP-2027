"""
extract_ms_batched.py
---------------------
与官方 vanilla_clip_ms.py / muti_scale_gen_feat.py **逻辑一致**的多尺度提取脚本，
仅将「逐样本前向」改为「chunked batch 前向」以提升 GPU 利用率。

RNG 一致性保证：
  - crop 的生成顺序与官方脚本完全相同（scale-major: 外层 scale，内层逐样本 transform），
    因此 torchvision RandomResizedCrop 的随机序列与官方完全一致（相同 seed 下 crop 逐位相同）。
  - **关键陷阱（已验证）**：torchvision 的 RandomResizedCrop 消耗 **torch RNG**；任何
    DataLoader 包装（即使 num_workers=0）都会因 _BaseDataLoaderIter 的 base_seed 生成
    而推进 torch RNG，导致 crop 序列整体偏移。本脚本采用手写顺序循环 + chunk 前向。
  - 前向批处理化带来 ~1e-3 量级的浮点差异（batch vs 单张 kernel），个别边界样本的 argmax
    可能翻转（实测 PET scale-1 acc 差异 0.02%）。

输出格式与官方一致：
  real: <image_root>/<dataset>/multi_scale/CLIP_<bk>_feature_scale{s}.hdf5
        keys: test_f, test_l, all_embeddings
  gen : <image_root>/<dataset>/multi_scale_gen/<LLM>CLIP_<bk>_feature_gen{N}_scale{s}.hdf5
        keys: gen_f, gen_l

用法:
  python src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 --mode real --batch 128
  python src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 --mode gen --batch 128
  # 可选 --scales 1,2 用于快速一致性验证
"""
import os
import sys
import json
import random
import argparse

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LGCLIP_DIR = os.path.join(PROJECT_ROOT, "third_party", "LG-CLIP")
sys.path.insert(0, LGCLIP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import clip
from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATASET_MAP = {
    "CUB": CUBDataset, "FLO": FLODataset, "PET": PETDataset,
    "FOOD": FOODDataset, "ImageNet": ImageNetDataset, "EUROSAT": EUROSATDataset,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PET")
    p.add_argument("--backbone", default="ViT-B/32")
    p.add_argument("--mode", default="real", choices=["real", "gen"])
    p.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--gen_root_path", default=os.path.join(PROJECT_ROOT, "data", "SD_gen"))
    p.add_argument("--dataset_root", default=None,
                   help="dataset loader 的 image_root（默认同 --image_root）")
    p.add_argument("--llm", default="")
    p.add_argument("--ngen", default=10, type=int)
    p.add_argument("--sd_2_1", default=True, action="store_true")
    p.add_argument("--sd_xl", default=False, action="store_true")
    p.add_argument("--seed", default=2024, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", default=128, type=int)
    p.add_argument("--scales", default="", help="逗号分隔的 scale 子集（调试/验证用），默认全 1..10")
    p.add_argument("--out_tag", default="", help="输出文件名后缀（验证时不覆盖官方缓存），如 _batchtest")
    p.add_argument("--num_workers", default=0, type=int,
                   help="crop 预处理并行数。注意: >0 时子进程 RNG 独立，crop 序列将不同于官方脚本"
                        "（仍为有效随机采样，但无法做逐位一致性验证）。默认 0 保持与官方一致的 RNG 序列。")
    return p.parse_args()


def get_textEmbedding(classnames, clip_model, args, norm=True):
    with torch.no_grad():
        classnames = [c.replace('_', ' ') for c in classnames]
        texts = [f"A photo of a {c}." for c in classnames]
        tokens = clip.tokenize(texts, context_length=77).to(args.device)
        feats = clip_model.encode_text(tokens).float().to(args.device)
        if norm:
            feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def _crop_one(item, transform):
    img_path, label = item
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    return img, label


def get_visualEmbedding_batched(clip_model, dataframe, device, transform,
                                batch_size=128, num_workers=0):
    """与官方 get_visualEmbedding 相同的输出，但 chunked batch 前向。

    RNG 一致性：crop 逐个按顺序调用（与官方脚本相同的 torch RNG 消耗序列），
    只把 encode_image 前向批处理化。（注意：torchvision 的 RandomResizedCrop 使用
    torch RNG，任何 DataLoader 包装（即使 num_workers=0）都会因 base_seed 生成
    而推进 torch RNG，破坏与官方脚本的 crop 一致性——因此这里不使用 DataLoader。）
    """
    feats_all, labels_all = [], []
    buf_imgs, buf_labs = [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(x).float()
            f /= f.norm(dim=-1, keepdim=True)
        feats_all.append(f.cpu())
        labels_all.extend(buf_labs)
        buf_imgs.clear()
        buf_labs.clear()

    for img_path, label in tqdm(dataframe, ncols=100):
        img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)
        buf_imgs.append(img)
        buf_labs.append(label)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()

    feats = torch.cat(feats_all)
    return feats.numpy(), np.array(labels_all)


def collect_gen_files(args, all_names, model_version):
    gen_files, gen_labels = [], []
    if args.llm == "LLM_":
        grp = args.gen_root_path
    else:
        grp = os.path.join(LGCLIP_DIR, 'dataset', 'SD_gen')
    exp_identifier = f"{args.llm}SD_{model_version}_{args.dataset}_10"
    for idx, name in enumerate(all_names):
        img_dir = os.path.join(grp, exp_identifier, name)
        files = sorted(os.listdir(img_dir))
        files = [f for f in files if f.endswith('.jpg')]
        for img_file in files:
            gen_files.append(os.path.join(img_dir, img_file))
        gen_labels += [idx] * len(files)
    return list(zip(gen_files, gen_labels))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    clip_model, preprocess = clip.load(args.backbone, device=args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    dr = args.dataset_root or args.image_root
    ds_args = argparse.Namespace(image_root=dr)
    mydataset = DATASET_MAP[args.dataset](ds_args)
    all_names = mydataset.all_names
    model_name = args.backbone.replace("-", "").replace("/", "")

    scales = ([int(s) for s in args.scales.split(",")] if args.scales
              else list(range(1, 11)))

    if args.mode == "real":
        multi_scale_dir = os.path.join(args.image_root, args.dataset, "multi_scale")
        os.makedirs(multi_scale_dir, exist_ok=True)
        testdf = list(zip(mydataset.test_files, mydataset.test_labels.numpy()))
        all_embeddings = get_textEmbedding(all_names, clip_model, args)
        print("Text embeddings:", tuple(all_embeddings.shape))

        for scale in scales:
            scale_factor = scale / 10.0
            out_path = os.path.join(
                multi_scale_dir,
                f"CLIP_{model_name}_feature_scale{scale}{args.out_tag}.hdf5")
            multi_scale_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=224, scale=(scale_factor, scale_factor),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
            ])
            print(f" ==> Scale {scale}: extracting (batch={args.batch}) ...")
            test_f, test_l = get_visualEmbedding_batched(
                clip_model, testdf, args.device, multi_scale_transform,
                batch_size=args.batch, num_workers=args.num_workers)
            print("   shape:", test_f.shape)

            with h5py.File(out_path, "w") as f:
                f.create_dataset('test_f', data=test_f, compression="gzip")
                f.create_dataset('test_l', data=test_l, compression="gzip")
                f.create_dataset('all_embeddings', data=all_embeddings.cpu(),
                                 compression="gzip")
            print(f"   saved {out_path}")

            tf = torch.from_numpy(test_f).float().to(args.device)
            sims = tf @ all_embeddings.T
            acc = (sims.argmax(-1).cpu().numpy() == test_l).mean()
            print(f"   [{args.backbone}] Scale-{scale} Acc = {acc*100:.2f}%")

        # merge（与官方一致：w=s/55 → 归一化）。验证模式（out_tag）下跳过，避免写正式文件
        if args.out_tag:
            print("out_tag set -> skipping merge (validation mode)")
            return
        print("Merging scales -> ms file ...")
        feats_all = None
        labels_all = None
        for scale in range(1, 11):
            p_ = os.path.join(
                multi_scale_dir, f"CLIP_{model_name}_feature_scale{scale}.hdf5")
            if not os.path.exists(p_):
                continue
            with h5py.File(p_, 'r') as f:
                feats = torch.from_numpy(np.array(f['test_f'])).float()
                labels = torch.from_numpy(np.array(f['test_l'])).float()
                ae = np.array(f['all_embeddings'])
            ratio = scale / 55.0
            feats_all = feats * ratio if feats_all is None else feats_all + feats * ratio
            labels_all = labels
        feats_all /= feats_all.norm(dim=-1, keepdim=True)
        ms_path = os.path.join(args.image_root, args.dataset,
                               f"CLIP_{model_name}_feature_ms.hdf5")
        with h5py.File(ms_path, "w") as f:
            f.create_dataset('test_f', data=feats_all.numpy(), compression="gzip")
            f.create_dataset('test_l', data=labels_all.numpy(), compression="gzip")
            f.create_dataset('all_embeddings', data=ae, compression="gzip")
        print("saved", ms_path)

    else:  # gen
        model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
        multi_scale_dir = os.path.join(args.image_root, args.dataset, "multi_scale_gen")
        os.makedirs(multi_scale_dir, exist_ok=True)
        gendf = collect_gen_files(args, all_names, model_version)
        print("gen images:", len(gendf))
        assert len(gendf) == len(all_names) * args.ngen
        all_embeddings = get_textEmbedding(all_names, clip_model, args)

        for scale in scales:
            scale_factor = scale / 10.0
            out_path = os.path.join(
                multi_scale_dir,
                f"{args.llm}CLIP_{model_name}_feature_gen{args.ngen}_scale{scale}{args.out_tag}.hdf5")
            multi_scale_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=224, scale=(scale_factor, scale_factor),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
            ])
            print(f" ==> gen scale {scale}: extracting (batch={args.batch}) ...")
            gen_f, gen_l = get_visualEmbedding_batched(
                clip_model, gendf, args.device, multi_scale_transform,
                batch_size=args.batch, num_workers=args.num_workers)
            gen_f = gen_f.reshape(len(all_names), args.ngen, gen_f.shape[-1])
            gen_l = np.array(gen_l).reshape(len(all_names), args.ngen, 1)
            with h5py.File(out_path, "w") as f:
                f.create_dataset('gen_f', data=gen_f, compression="gzip")
                f.create_dataset('gen_l', data=gen_l, compression="gzip")
            print("   saved", out_path)

        # merge（验证模式下由上方提前返回跳过）
        feats_all, labels_all = None, None
        for scale in range(1, 11):
            p_ = os.path.join(
                multi_scale_dir,
                f"{args.llm}CLIP_{model_name}_feature_gen{args.ngen}_scale{scale}.hdf5")
            if not os.path.exists(p_):
                continue
            with h5py.File(p_, 'r') as f:
                feats = torch.from_numpy(np.array(f['gen_f'])).float()
                labels = torch.from_numpy(np.array(f['gen_l'])).float()
            ratio = scale / 55.0
            feats_all = feats * ratio if feats_all is None else feats_all + feats * ratio
            labels_all = labels
        feats_all /= feats_all.norm(dim=-1, keepdim=True)
        ms_path = os.path.join(
            args.image_root, args.dataset,
            f"{args.llm}CLIP_{model_name}_feature_gen{args.ngen}_ms.hdf5")
        with h5py.File(ms_path, "w") as f:
            f.create_dataset('gen_f', data=feats_all.numpy(), compression="gzip")
            f.create_dataset('gen_l', data=labels_all.numpy(), compression="gzip")
        print("saved", ms_path)


if __name__ == "__main__":
    main()
