"""
build_eurosat.py
----------------
从 HuggingFace blanchon/EuroSAT_RGB parquet 重建 EuroSAT 图像到
data/EUROSAT/2750/<ClassName>/，并构建 split_EuroSAT.json（LG-CLIP 兼容格式）。

规则:
  - test  = 全部 27000 张（论文 §4.1 "use all the original images as the test set"）
  - train = 每类前 5 张（保证 all_names 覆盖全部类别；train 特征仅用于缓存）

用法:
  python scripts/setup/build_eurosat.py
"""
import os
import io
import json
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DL = os.path.join(PROJECT_ROOT, "data", "_downloads")
OUT_BASE = os.path.join(PROJECT_ROOT, "data", "EUROSAT")
IMG_ROOT = os.path.join(OUT_BASE, "2750")
OUT_JSON = os.path.join(OUT_BASE, "split_EuroSAT.json")

CLASS_NAMES = {
    0: "Annual_Crop",
    1: "Forest",
    2: "Herbaceous_Vegetation",
    3: "Highway",
    4: "Industrial_Buildings",
    5: "Pasture",
    6: "Permanent_Crop",
    7: "Residential_Buildings",
    8: "River",
    9: "SeaLake",
}
N_TRAIN_PER_CLASS = 5

def main():
    os.makedirs(IMG_ROOT, exist_ok=True)
    counters = defaultdict(int)
    records = []  # (rel_path, label, class_name)

    for split in ["train", "test", "validation"]:
        path = os.path.join(DL, f"eurosat_{split}.parquet")
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        print(f"Reading {split}: {n} rows")
        table = pf.read()
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for img, lab in zip(images, labels):
            cname = CLASS_NAMES[lab]
            cdir = os.path.join(IMG_ROOT, cname)
            os.makedirs(cdir, exist_ok=True)
            idx = counters[cname]
            counters[cname] += 1
            fname = f"{cname}_{idx:05d}.png"
            b = img["bytes"]
            # 保存为 PNG（无损），验证可解码
            im = Image.open(io.BytesIO(b)).convert("RGB")
            im.save(os.path.join(cdir, fname))
            records.append([f"{cname}/{fname}", lab, cname])

    print("Total images:", len(records))
    per_class = defaultdict(int)
    for _, lab, cname in records:
        per_class[cname] += 1
    print("Per class:", dict(per_class))

    # 排序输出（按 class label, 文件名）保证可复现
    records = sorted(records, key=lambda r: (r[1], r[0]))
    test = [r for r in records]
    train = []
    seen = defaultdict(int)
    for r in records:
        if seen[r[1]] < N_TRAIN_PER_CLASS:
            train.append(r)
            seen[r[1]] += 1

    with open(OUT_JSON, "w") as fp:
        json.dump({"train": train, "test": test}, fp, indent=1)
    print(f"Saved {OUT_JSON}: train={len(train)}, test={len(test)}")

    # 一致性检查
    all_names = [' '.join(n.split('_')) for n in sorted(set(d[2] for d in train))]
    for d in test[:50]:
        assert all_names[d[1]] == ' '.join(d[2].split('_')), (d, all_names[d[1]])
    print("Label/all_names consistency OK:", all_names)

if __name__ == "__main__":
    main()
