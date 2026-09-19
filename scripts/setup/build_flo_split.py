"""
build_flo_split.py
------------------
构建 LG-CLIP 兼容的 FLO split 文件: data/FLO/Flowers102/split_OxfordFlowers.json

规则（与 LG-CLIP 论文 §4.1 一致）:
  - test  = 数据集的全部图像（8189 张）
  - train = 每类前 5 张（保证非空；train 特征仅用于缓存，不参与评估）

格式: {"train": [[rel_path, label], ...], "test": [[rel_path, label], ...]}
label = imagelabels.mat 的 1-based 标签 - 1
（与 FLODataset.all_names = [cat_to_name[str(i)] for i in 1..102] 的 0-based 索引对齐）

用法:
  python scripts/setup/build_flo_split.py
"""
import os
import json
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(PROJECT_ROOT, "data", "FLO", "Flowers102")
JPG_DIR = os.path.join(BASE, "jpg")
OUT = os.path.join(BASE, "split_OxfordFlowers.json")
N_TRAIN_PER_CLASS = 5

def main():
    labels = loadmat(os.path.join(BASE, "imagelabels.mat"))["labels"].squeeze()  # (8189,), 1-based
    files = sorted(os.listdir(JPG_DIR))
    assert len(files) == len(labels), (len(files), len(labels))

    per_class = defaultdict(list)
    for f, lab in zip(files, labels):
        per_class[int(lab) - 1].append(f)
    print(f"{len(files)} images, {len(per_class)} classes")

    train, test = [], []
    for c in sorted(per_class.keys()):
        fs = sorted(per_class[c])
        for f in fs:
            test.append([f, c])
        for f in fs[:N_TRAIN_PER_CLASS]:
            train.append([f, c])

    with open(OUT, "w") as fp:
        json.dump({"train": train, "test": test}, fp, indent=1)
    print(f"Saved {OUT}: train={len(train)}, test={len(test)}")

    # 一致性检查
    cat = json.load(open(os.path.join(BASE, "cat_to_name.json")))
    all_names = [cat[str(i)] for i in range(1, 103)]
    assert len(all_names) == 102
    print("OK. sample names:", all_names[0], "|", all_names[20], "|", all_names[-1])

if __name__ == "__main__":
    main()
