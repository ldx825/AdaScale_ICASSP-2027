"""
verify_batch_equiv.py
---------------------
验证 extract_ms_batched.py 的输出与官方脚本特征的一致性。

用法（PET 示例）:
  # 1) 先用 batch 版重跑 scale 1（不覆盖官方文件）:
  python src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 \
      --mode real --scales 1 --batch 128 --out_tag _batchtest
  # 2) 对比:
  python scripts/audit/verify_batch_equiv.py --dataset PET --backbone ViT-B/32 --scale 1

判定标准:
  - crop RNG 序列一致 ⇒ 特征逐位接近（max|diff| < 1e-4 允许 fp 舍入）
"""
import os
import sys
import argparse

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PET")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--scale", default=1, type=int)
    ap.add_argument("--image_root", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--key", default="test_f")
    args = ap.parse_args()

    bk = args.backbone.replace("-", "").replace("/", "")
    d = os.path.join(args.image_root, args.dataset, "multi_scale")
    f_off = os.path.join(d, f"CLIP_{bk}_feature_scale{args.scale}.hdf5")
    f_bat = os.path.join(d, f"CLIP_{bk}_feature_scale{args.scale}_batchtest.hdf5")
    assert os.path.exists(f_off), f"missing {f_off}"
    assert os.path.exists(f_bat), f"missing {f_bat}"

    with h5py.File(f_off, "r") as f:
        a = np.array(f[args.key])
    with h5py.File(f_bat, "r") as f:
        b = np.array(f[args.key])
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = np.abs(a - b)
    print(f"scale {args.scale}: shape={a.shape}")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}")
    print(f"  frac(>1e-4) = {(diff > 1e-4).mean():.6f}")
    # 行级最大差异分布（找出不一致的样本，若有）
    row_max = diff.max(axis=1)
    print(f"  rows with max|diff|>1e-4: {(row_max > 1e-4).sum()} / {len(row_max)}")
    if diff.max() < 1e-4:
        print("  VERDICT: batch 版与官方版一致（crop RNG 序列相同，仅前向批处理化）")
    else:
        idx = np.argsort(row_max)[-5:]
        print(f"  top-5 mismatch rows: {idx.tolist()} (their max diffs: {row_max[idx].tolist()})")
        print("  VERDICT: 存在不一致，需检查 RNG/顺序")


if __name__ == "__main__":
    main()
