"""
runtime_bench.py
----------------
真实运行时的动态多尺度执行基准（active-set batching）：

  所有样本从 scale 1 开始活跃；每完成一个 scale，策略决定哪些样本退出；
  退出的样本不参与后续 scale 的 crop/forward（真实节省计算）。

对照配置:
  - full-10: 全部样本跑 10 个 scale
  - fixed-k: 全部样本跑 k 个 scale
  - adascale: 策略驱动（退出即停止）

测量:
  - 总耗时（wall-clock，torch.cuda.synchronize 同步）
  - 每尺度处理量（各 scale 的 forward 数）
  - images/s、平均 scales

用法（需要 GPU）:
  python src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode adascale \
      --policy stab --r 3 --delta 0.35 --batch 16 --warmup 2 --repeats 3
"""
import os
import sys
import json
import time
import random
import argparse
from dataclasses import asdict

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
from utils.myDataset import (  # noqa: E402
    CUBDataset, FLODataset, PETDataset, FOODDataset, ImageNetDataset, EUROSATDataset,
)
from src.adascale.simulate import (  # noqa: E402
    StabilityPolicy, FixedKPolicy, MarginOnlyPolicy, LOGIT_TAU,
)

DATASET_MAP = {"CUB": CUBDataset, "FLO": FLODataset, "PET": PETDataset,
               "FOOD": FOODDataset, "ImageNet": ImageNetDataset, "EUROSAT": EUROSATDataset}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PET")
    p.add_argument("--backbone", default="ViT-B/32")
    p.add_argument("--mode", default="adascale", choices=["full", "fixed", "adascale", "single"])
    p.add_argument("--policy", default="stab", choices=["stab", "margin"])
    p.add_argument("--k", default=5, type=int)          # fixed 模式
    p.add_argument("--r", default=3, type=int)
    p.add_argument("--delta", default=0.35, type=float)
    p.add_argument("--eps", default=0.01, type=float)
    p.add_argument("--min_scale", default=1, type=int)
    p.add_argument("--batch", default=16, type=int)
    p.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", default=1, type=int)
    p.add_argument("--repeats", default=3, type=int)
    p.add_argument("--max_samples", default=0, type=int, help="调试用，限制样本数")
    p.add_argument("--seed", default=2024, type=int)
    p.add_argument("--out", default="")
    return p.parse_args()


def build_transform(scale_factor):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(scale_factor, scale_factor),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])


@torch.no_grad()
def run_once(model, img_paths, policy, args, labels_game=False):
    """执行一次完整推理流程，返回 (pred, scales_used, time_seconds, per_scale_counts)。"""
    N = len(img_paths)
    device = args.device
    active = np.arange(N)
    prefix_sum = torch.zeros(N, 512, device=device)  # CLIP feature dim（ViT-B/32: 512）
    pred_final = np.full(N, -1, dtype=np.int64)
    scales_used = np.zeros(N, dtype=np.int64)
    top1_hist = np.full((N, 11), -1, dtype=np.int64)
    prev_prob = None
    per_scale_counts = {}

    # 预打开图像（避免每 scale 重复 IO 开销被算入——注意：官方脚本每 scale 重新读图，
    # 这里为公平也保持"每 scale 重新打开"，但我们将它放在计时内）
    t0 = time.time()
    scale_list = getattr(args, "scale_list", list(range(1, 11)))
    for step, t in enumerate(scale_list, start=1):
        if len(active) == 0:
            break
        per_scale_counts[t] = len(active)
        sf = t / 10.0
        tf = build_transform(sf)

        feats = []
        for start in range(0, len(active), args.batch):
            idx = active[start:start + args.batch]
            imgs = []
            for i in idx:
                img = Image.open(img_paths[i]).convert("RGB")
                imgs.append(tf(img))
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x).float()
            f /= f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        f_all = torch.cat(feats).to(device)

        prefix_sum[active] = prefix_sum[active] + float(t) * f_all
        f_norm = prefix_sum[active] / prefix_sum[active].norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if policy is not None and needs_logits(policy):
            logits = f_norm @ policy_prototypes[0].to(device).T
            prob = torch.softmax(logits / LOGIT_TAU, dim=-1)
            top1 = logits.argmax(-1).cpu().numpy()
            top1_hist[active, t] = top1
            pred_final[active] = top1
        else:
            top1 = np.zeros(len(active), dtype=np.int64)  # 占位（无 prototypes 时无法预测）

        scales_used[active] = step

        if policy is not None:
            if t == 1 or prev_prob is None:
                js_prev = torch.zeros(len(active), device=device)
            else:
                m = 0.5 * (prob + prev_prob[active])
                eps_ = 1e-12
                js_prev = 0.5 * (prob * torch.log(prob.clamp_min(eps_) / m.clamp_min(eps_))).sum(-1) \
                          + 0.5 * (prev_prob[active] * torch.log(prev_prob[active].clamp_min(eps_) / m.clamp_min(eps_))).sum(-1)
            prev_prob = torch.zeros(N, prob.shape[-1], device=device) if prev_prob is None else prev_prob
            prev_prob[active] = prob
            exit_mask = policy.decide(t, active, f_norm, logits, prob, js_prev, top1_hist)
            active = active[~exit_mask]
    torch.cuda.synchronize()
    t1 = time.time()
    # 注意所有样本最终使用的 scale 数（在上面的循环里已记录）
    return pred_final, scales_used, (t1 - t0), per_scale_counts


policy_prototypes = [None]
needs_logits_cache = {}


def needs_logits(policy):
    return True


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    ds_args = argparse.Namespace(image_root=args.image_root)
    mydataset = DATASET_MAP[args.dataset](ds_args)
    img_paths = list(mydataset.test_files)
    labels = mydataset.test_labels.numpy()
    if args.max_samples:
        img_paths = img_paths[:args.max_samples]
        labels = labels[:args.max_samples]
    N = len(img_paths)

    model, _ = clip.load(args.backbone, device=args.device)
    model.eval()

    # prototypes：text prototypes（从缓存读，或重算）
    import h5py
    from src.utils.load_features import real_scale_path
    with h5py.File(real_scale_path(args.image_root, args.dataset, args.backbone, 1), "r") as f:
        text_emb = torch.from_numpy(np.array(f["all_embeddings"])).float()
    policy_prototypes[0] = text_emb

    # 构造策略
    if args.mode == "full":
        args.scale_list = list(range(1, 11))
        policy = FixedKPolicy(10)
    elif args.mode == "single":
        # 仅评估最大 scale（Single 模式；crop 为真实执行的独立 RNG 流）
        args.scale_list = [10]
        policy = FixedKPolicy(10)
    elif args.mode == "fixed":
        policy = FixedKPolicy(args.k)
    else:
        if args.policy == "stab":
            policy = StabilityPolicy(r=args.r, delta=args.delta, eps=args.eps,
                                     min_scale=args.min_scale, use_js=(args.eps < 1.0),
                                     use_margin=True)
        else:
            policy = MarginOnlyPolicy(delta=args.delta, min_scale=args.min_scale)

    print(f"[bench] {args.dataset}/{args.backbone} mode={args.mode} N={N} batch={args.batch}")
    for w in range(args.warmup):
        run_once(model, img_paths, policy, args)
        print(f"  warmup {w+1} done")

    times, accs, avg_scales_all = [], [], []
    for rep in range(args.repeats):
        pred, su, dt, psc = run_once(model, img_paths, policy, args)
        acc = (pred == labels).mean()
        times.append(dt); accs.append(acc); avg_scales_all.append(su.mean())
        print(f"  rep {rep+1}: time={dt:.2f}s acc={acc*100:.2f}% avg_scales={su.mean():.2f}")

    result = {
        "dataset": args.dataset, "backbone": args.backbone, "mode": args.mode,
        "policy": (vars(policy) if hasattr(policy, "__dict__") else str(policy)),
        "N": N, "batch": args.batch,
        "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
        "throughput_ips": float(N / np.mean(times)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "avg_scales": float(np.mean(avg_scales_all)),
        "per_scale_counts": {str(k): int(v) for k, v in psc.items()},
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
