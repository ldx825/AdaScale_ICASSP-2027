# Reproducibility — AdaScale-CLIP

> 项目根: `/root/autodl-tmp/EviZO-VP/AdaScale-CLIP`（所有文件均位于项目内，见 §1 隔离规则）

## 1. 环境

| 项 | 值 |
|---|---|
| OS / 容器 | AutoDL 容器（16 core / 62GB RAM / RTX 4080 SUPER 32GB） |
| Python | 3.12.3（miniconda3 系统解释器） |
| venv | `$PROJECT_ROOT/.venv`（`--system-site-packages`，复用系统 torch 2.8.0+cu128） |
| CUDA | driver 13.2，torch cuda 12.8 build |
| CLIP 权重缓存 | `$PROJECT_ROOT/.home/.cache/clip/`（HOME 重定向） |
| pip cache | `$PROJECT_ROOT/.cache/pip` |

**隔离措施（全部生效）**：
```bash
export HOME="$PROJECT_ROOT/.home"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.tmp"
```

## 2. Baseline 代码

- repo: `third_party/LG-CLIP`，commit `e0a3c6fe3bfd4ffaaec39674f7ecc5e1e0fb73a5`
- repo 修复：补入 `clip/bpe_simple_vocab_16e6.txt.gz`（官方 repo 缺失，见 `notes/lgclip_instrumentation.patch.md`）
- 无其他源码改动。

## 3. 数据集准备（全部位于 `data/`）

| 数据集 | 来源 | 本地路径 | test 数 | 说明 |
|---|---|---|---|---|
| PET | Oxford-IIIT Pets 官方 tar | `data/PET/OxfordPets/{images,annotations}` | 7390（全部） | split 由 `scripts/setup/build_pet_split.py` 生成 |
| EUROSAT | HF `blanchon/EuroSAT_RGB` parquet（官方 27000 张 RGB） | `data/EUROSAT/2750/<Class>/` | 27000（全部） | 重建+split 由 `scripts/setup/build_eurosat.py` |
| FLO | Oxford 102 Flowers 官方 tar + labels.mat | `data/FLO/Flowers102/{jpg,cat_to_name.json}` | 8189（全部） | split 由 `scripts/setup/build_flo_split.py` |

**split 约定（与 LG-CLIP 论文 §4.1 一致）**：
- `test` = 数据集全部图像（论文明确 "use all the original images as the test set"）
- `train` = 每类 5 张（仅用于满足官方 loader 的非空要求与缓存；**不参与任何评估**）
- 官方 repo **未提供** `split_*.json`；本项目的构建脚本已记录于 `scripts/setup/`，label 与 `all_names` 的字母序严格一致。

## 4. 运行 baseline

```bash
# 单尺度（text prototype zero-shot，对应论文 Table 1 的 CLIP 列）
bash scripts/baseline/run_baseline.sh single PET ViT-B/32
# 多尺度（10 scales + triangular 融合；per-scale acc 打印）
bash scripts/baseline/run_baseline.sh ms PET ViT-B/32
```

产物：
- `data/<dataset>/multi_scale/CLIP_<bk>_feature_scale{1..10}.hdf5`
- `data/<dataset>/CLIP_<bk>_feature_ms.hdf5`
- 日志 `outputs/baseline/<dataset>/*.log`

## 5. Scale Utility Audit

```bash
python scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/32 --proto text
python src/analysis/plot_audit.py     --dataset PET --backbone ViT-B/32 --proto text
```

产物：
- `outputs/scale_audit/<dataset>/<bk>/per_sample_<proto>.parquet`
- `outputs/scale_audit/<dataset>/<bk>/summary_<proto>.json`
- `figures/audit/<dataset>_<bk>_<proto>_fig{A..F}.{pdf,png}`

**一致性协议**：`run_scale_audit.py` 首先验证 `prefix-10 聚合特征 == 官方 *_feature_ms.hdf5`（要求 max|diff| < 1e-4；实测见输出），确保 audit 未改变 baseline 数值。

## 6. 已知偏离（诚实记录）

1. **LLM prompt 路径不可用**：`prompts_gen.py` 需要 xAI Grok API key，本机无凭据。
   → 生成图走 plain-SD 路径（`sd_gen.py`，class-label prompt）。论文 Table 3 显示 MS 增益独立于 LLM（CLIP&MS vs CLIP），本项目研究核心（尺度冗余）不依赖 LLM 路径。
2. **SD 生成图与作者的 exact 生成集不保证一致**（diffusers 版本/权重来源差异）；固定官方 seed=2024 与官方参数。可能造成与 Table 1 "Ours" 数字的小差异——**以可复现路径为准**，并在 `outputs/baseline/reproduction.csv` 记录差异。
3. split json 为自建（见 §3）。test 用全量图像符合论文描述；预测类别与 label 内部一致（已断言检查）。

## 7. 硬件与随机性

- 全局 seed=2024（官方默认）；提取与审计脚本固定 seed；`cudnn.benchmark=True`
- 多尺度 crop 的随机性：`RandomResizedCrop(scale=(sf,sf))` 位置/ratio 随机；
  torchvision 的 RRC **消费 Torch RNG**（不是 Python random）。seed 固定、完整
  顺序执行时可与官方逐位复现。
- **关键陷阱（已修复）**：任何 DataLoader 包装（即使 `num_workers=0`）都会因
  `_BaseDataLoaderIter` 的 base_seed 生成而额外消费 Torch RNG，导致 crop 序列
  与官方不一致——实测 batch 版特征与官方最大偏差达 0.347（全行）。诊断见
  `logs/diag_rng*.log`。
- **修复方案**：`src/analysis/extract_ms_batched.py` 完全禁用 DataLoader——手动
  逐样本 crop（严格保持官方 RNG 顺序），按 chunk（默认 128）栈批前向。验证：
  - 与官方逐样本流程等价：max|diff| ≈ 1.9e-3（浮点级），acc 偏差 0.02%
  - audit 一致性协议：prefix-10 聚合 vs 官方 `*_feature_ms.hdf5`，
    max|diff| = 2.4e-7（< 1e-4 阈值）——正式审计基于官方缓存数值
- 运行时基准（`src/adascale/runtime_bench.py`）：batch=32，warmup=2，
  重复 3 次取均值；计时用 `torch.cuda.synchronize()` 包裹。
- AdaScale 评估全部在预提取的 per-scale 特征缓存上进行（同一 crop 池、配对
  公平）；早停模拟与 dynamic execution 语义一致（退出样本不再计算后续 scale，
  最终预测 = 退出时的 prefix 融合预测）。
