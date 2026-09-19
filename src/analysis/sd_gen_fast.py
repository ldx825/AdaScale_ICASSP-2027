"""
sd_gen_fast.py
--------------
与官方 sd_gen.py **生成参数严格一致**的生成脚本（仅将 pipeline 加载改为一次，
避免官方逐类重复加载模型的巨大时间开销）。

一致性要点（与官方 sd_gen.py 对比）:
  - 模型: stabilityai/stable-diffusion-2-1-base（本地镜像 Manojb/...，diffusers 格式）
  - prompt: f"A photo of a {class_name}."（class_name 已 change_form）
  - 每张图 i: generator.manual_seed(i)（官方行为）
  - 输出: <gen_root>/SD_2.1_<dataset>_10/<class_name>/{i}_{class_name}.jpg
  - 全局 seed=2024；device cuda:0

用法:
  python src/analysis/sd_gen_fast.py --dataset PET --ngen 10
  python src/analysis/sd_gen_fast.py --dataset EUROSAT --ngen 10
"""
import os
import re
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LGCLIP_DIR = os.path.join(PROJECT_ROOT, "third_party", "LG-CLIP")
sys.path.insert(0, LGCLIP_DIR)

from utils.helper_func import numpy_to_pil  # noqa: E402
from utils.myDataset import (  # noqa: E402
    CUBDataset, FLODataset, PETDataset, FOODDataset, ImageNetDataset, EUROSATDataset,
)

DATASET_MAP = {
    "CUB": CUBDataset, "FLO": FLODataset, "PET": PETDataset,
    "FOOD": FOODDataset, "ImageNet": ImageNetDataset, "EUROSAT": EUROSATDataset,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PET")
    p.add_argument("--ngen", default=10, type=int)
    p.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--gen_root_path", default=os.path.join(PROJECT_ROOT, "data", "SD_gen"))
    p.add_argument("--model_id", default="Manojb/stable-diffusion-2-1-base",
                   help="本地缓存的 SD 2.1 diffusers 镜像（原始 stabilityai repo 已 gated）")
    p.add_argument("--seed", default=2024, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_inference_steps", default=50, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from diffusers import StableDiffusionPipeline

    ds_args = argparse.Namespace(image_root=args.image_root)
    mydataset = DATASET_MAP[args.dataset](ds_args)
    all_names = mydataset.all_names
    exp_identifier = f"SD_2.1_{args.dataset}_{args.ngen}"
    out_root = os.path.join(args.gen_root_path, exp_identifier)
    print(f"[gen] dataset={args.dataset} classes={len(all_names)} Ngen={args.ngen} -> {out_root}")

    print("[gen] loading pipeline (once) ...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id, safety_checker=None, torch_dtype=torch.float32
    ).to(args.device)
    print("[gen] pipeline ready.")

    for class_label, class_name in enumerate(all_names):
        img_dir = os.path.join(out_root, class_name)
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        prompt = f"A photo of a {class_name}."
        existing = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        done_idx = set()
        for f in existing:
            m = re.match(r'^(\d+)_', f)
            if m:
                done_idx.add(int(m.group(1)))
        todo = [i for i in range(args.ngen) if i not in done_idx]
        print(f"[gen] [{class_label+1}/{len(all_names)}] '{class_name}': "
              f"{len(done_idx)} existing, generating {len(todo)} ...")
        for i in todo:
            generator = torch.Generator(device=args.device).manual_seed(i)
            image_out = pipeline(prompt, output_type="pt", generator=generator,
                                 num_inference_steps=args.num_inference_steps)[0]
            img_path = os.path.join(img_dir, f"{i}_{class_name}.jpg")
            numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(
                img_path, "JPEG")
        print(f"[gen]   done '{class_name}'")

    total = sum(len([f for f in os.listdir(os.path.join(out_root, c)) if f.endswith('.jpg')])
                for c in all_names)
    print(f"[gen] COMPLETE. total images: {total} (expected {len(all_names)*args.ngen})")


if __name__ == "__main__":
    main()
