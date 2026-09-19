"""
load_features.py
----------------
加载 LG-CLIP per-scale HDF5 缓存的工具函数。

目录约定（与官方脚本一致）:
  real: <image_root>/<dataset>/multi_scale/CLIP_<bk>_feature_scale{s}.hdf5
        keys: test_f (N,D) float32 L2-normalized, test_l (N,), all_embeddings (C,D)
  gen : <image_root>/<dataset>/multi_scale_gen/<LLM>CLIP_<bk>_feature_gen{N}_scale{s}.hdf5
        keys: gen_f (C,Ngen,D), gen_l (C,Ngen,1)
"""
import os
import re
import h5py
import numpy as np
import torch

NUM_SCALES = 10
SCALE_WEIGHTS = np.arange(1, NUM_SCALES + 1, dtype=np.float64)  # w_s = s (相对权重)


def backbone_name(backbone: str) -> str:
    return backbone.replace("-", "").replace("/", "")


def real_scale_path(image_root, dataset, backbone, scale):
    return os.path.join(
        image_root, dataset, "multi_scale",
        f"CLIP_{backbone_name(backbone)}_feature_scale{scale}.hdf5")


def real_ms_path(image_root, dataset, backbone):
    return os.path.join(
        image_root, dataset, f"CLIP_{backbone_name(backbone)}_feature_ms.hdf5")


def gen_scale_path(image_root, dataset, backbone, llm, ngen, scale):
    return os.path.join(
        image_root, dataset, "multi_scale_gen",
        f"{llm}CLIP_{backbone_name(backbone)}_feature_gen{ngen}_scale{scale}.hdf5")


def gen_ms_path(image_root, dataset, backbone, llm, ngen):
    return os.path.join(
        image_root, dataset,
        f"{llm}CLIP_{backbone_name(backbone)}_feature_gen{ngen}_ms.hdf5")


def load_real_scales(image_root, dataset, backbone, scales=range(1, NUM_SCALES + 1),
                     dtype=torch.float32):
    """返回 dict[scale] = (test_f, ) 与 labels、text_embeddings。

    Returns:
        feats:  list[Tensor(N,D)]  按 scale 顺序（1..10）
        labels: Tensor(N,)
        text_embeddings: Tensor(C,D)（每个 scale 文件里相同，取第一个的）
    """
    feats, labels, text_emb = [], None, None
    for s in scales:
        p = real_scale_path(image_root, dataset, backbone, s)
        with h5py.File(p, "r") as f:
            feats.append(torch.from_numpy(np.array(f["test_f"])).to(dtype))
            if labels is None:
                labels = torch.from_numpy(np.array(f["test_l"]))
                text_emb = torch.from_numpy(np.array(f["all_embeddings"])).to(dtype)
    return feats, labels, text_emb


def load_gen_scales(image_root, dataset, backbone, llm, ngen,
                    scales=range(1, NUM_SCALES + 1), dtype=torch.float32):
    """返回 gen 特征: list[Tensor(C,Ngen,D)], labels(C,Ngen,1)。"""
    feats, labels = [], None
    for s in scales:
        p = gen_scale_path(image_root, dataset, backbone, llm, ngen, s)
        with h5py.File(p, "r") as f:
            feats.append(torch.from_numpy(np.array(f["gen_f"])).to(dtype))
            if labels is None:
                labels = np.array(f["gen_l"])
    return feats, labels


def prefix_aggregate(feats, t, weights=SCALE_WEIGHTS):
    """prefix 聚合: f_t = sum_{s<=t} w_s * f_s, 再 L2-normalize。

    Args:
        feats: list[Tensor(N,D)]（第 i 个对应 scale=i+1）
        t: 1-based 使用的 scale 数量
    Returns:
        Tensor(N,D) L2-normalized
    """
    acc = None
    for i in range(t):
        w = float(weights[i])
        acc = feats[i] * w if acc is None else acc + feats[i] * w
    return acc / acc.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def full_scale_aggregate(feats, weights=SCALE_WEIGHTS):
    """全 scale 聚合（官方 triangular 的等价形式）。"""
    return prefix_aggregate(feats, len(feats), weights)
