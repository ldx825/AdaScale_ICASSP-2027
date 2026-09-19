# 机器已有资源盘点（先查再下，避免重复下载）

> 盘点日期：2026-09-11（每次新增下载前先查阅本文件并复核）

## 1. 公共数据盘 `/root/autodl-pub`（7.3T 远程挂载，只读使用）

| 内容 | 状态 | 我们的用途 |
|---|---|---|
| **CUB200-2011**（`CUB200-2011.tgz` + `segmentations.tgz`） | ✅ 已有 | Tier-2 数据集直接解压使用，**不再下载** |
| ImageNet / ImageNet100 | ✅ 已有 | 若资源允许做 ImageNet 实验时使用（解压耗时大，先记录） |
| cifar-10/100, VOC, COCO 等 | ✅ | 与本项目无关 |
| PET / EUROSAT / FLO / CLIP 权重 / SD 模型 | ❌ 无 | 已自行下载（见下） |

## 2. 系统缓存

| 位置 | 内容 | 结论 |
|---|---|---|
| `/root/.cache/clip` | 空 | 无 CLIP 权重 |
| `/root/.cache/huggingface` | 空 | 无 HF 模型 |
| `/root/.cache/torch/hub/checkpoints` | 仅 resnet18 | 无可用权重 |
| `/root/.cache/modelscope` | 不存在 | — |
| `/root/autodl-tmp/data` | 仅 MNIST | 无关 |

## 3. 其他项目（只读参考）

- `EviZO-VP/third_party/BlackVIP`、`VP-LLR`：无 CLIP 权重、无 SD 模型目录。
- 全盘 `find`：无 `ViT-B*32.pt / ViT-B-16.pt / ViT-L-14.pt / RN50.pt`、无 stable-diffusion 模型目录。

## 4. 本项目已下载资源（唯一副本在项目内）

| 资源 | 位置 | 状态 |
|---|---|---|
| PET 数据集（7390 张） | `data/PET/OxfordPets/` | ✅ 完成 |
| EuroSAT 数据集（27000 张） | `data/EUROSAT/2750/` | ✅ 完成（HF parquet 重建） |
| FLO 数据集（8189 张） | `data/FLO/Flowers102/` | ✅ 完成 |
| CUB-200-2011（11788 张） | `data/CUB/CUB_200_2011/`（从公开盘解压） | ✅ 完成 |
| CLIP ViT-B/32 权重 | `.home/.cache/clip/ViT-B-32.pt` | ✅ 完成 |
| CLIP ViT-B/16 权重（2026-09-11 下载，350,837,078 B） | `.home/.cache/clip/ViT-B-16.pt` | ✅ 完成（断点续传，字节数精确匹配；加载验证通过） |
| SD 2.1 base（diffusers 格式） | `.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base` | ✅ 完成（官方 stabilityai repo 返回 401 gated，改用 Manojb 镜像） |

## 5. 待用资源（需要时再取，勿提前下载）

- CUB200-2011：`/root/autodl-pub/CUB200-2011/CUB_200_2011.tgz` → 解压到 `data/CUB/`
- 未来其他 CLIP backbone（ViT-B/16, ViT-L/14）：下载前先查本文件与系统缓存
