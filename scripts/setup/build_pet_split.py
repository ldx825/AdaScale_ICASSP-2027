"""
build_pet_split.py
------------------
构建 LG-CLIP 兼容的 PET split 文件: data/PET/OxfordPets/split_OxfordPets.json

规则（与 LG-CLIP 论文 §4.1 一致）:
  - test  = 数据集的全部图像（"use all the original images as the test set"）
  - train = 每类前 N 张（保证 myDataset.PETDataset 中 all_names 覆盖全部类别；
             train 特征在 LG-CLIP 流程中仅用于缓存，从不参与评估）

格式:
  {"train": [[rel_path, label, class_name], ...],
   "test":  [[rel_path, label, class_name], ...]}
  label 与 sorted(set(class_names)) 的字母序索引一致。

用法:
  python scripts/setup/build_pet_split.py
"""
import os
import re
import json
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(PROJECT_ROOT, "data", "PET", "OxfordPets")
IMG_DIR = os.path.join(BASE, "images")
OUT = os.path.join(BASE, "split_OxfordPets.json")

N_TRAIN_PER_CLASS = 5

def class_name_of(stem: str) -> str:
    # e.g. "american_bulldog_123" -> "american_bulldog"; "Abyssinian_5" -> "Abyssinian"
    return re.sub(r"_\d+$", "", stem)

def main():
    jpgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    per_class = defaultdict(list)
    for f in jpgs:
        stem = os.path.splitext(f)[0]
        per_class[class_name_of(stem)].append(f)

    classes = sorted(per_class.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(jpgs)} jpg images, {len(classes)} classes")

    train, test = [], []
    for c in classes:
        files = sorted(per_class[c])
        idx = class_to_idx[c]
        for f in files:
            test.append([f, idx, c])
        for f in files[:N_TRAIN_PER_CLASS]:
            train.append([f, idx, c])

    with open(OUT, "w") as fp:
        json.dump({"train": train, "test": test}, fp, indent=1)
    print(f"Saved {OUT}: train={len(train)}, test={len(test)}")

    # 验证: label 与 all_names 的一致性
    all_names = [' '.join(n.split('_')) for n in sorted(set(d[2] for d in train))]
    assert len(all_names) == len(classes)
    for d in test[:10]:
        assert all_names[d[1]] == ' '.join(d[2].split('_'))
    print("Label/all_names consistency OK. Classes:")
    for i, c in enumerate(classes):
        print(f"  {i:2d} {c}")

if __name__ == "__main__":
    main()
