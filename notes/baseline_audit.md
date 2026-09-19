# LG-CLIP Baseline Audit（ICASSP 2026）

> 审计对象：Dianxing Shi et al., "Can Synthetic Images Serve as Effective and Efficient Class Prototypes?", ICASSP 2026.
> Paper: arXiv:2512.17160v2；Code: https://github.com/DianxingShi/LG-CLIP（commit `e0a3c6fe3bfd4ffaaec39674f7ecc5e1e0fb73a5`）
> 审计日期：2026-09-11。所有结论以**代码为准**，与论文不一致处显式标注。

---

## A. Problem（论文动机）

1. **VLM 依赖 image-text pair**：CLIP 等模型的零样本分类依赖 text encoder 与 prompt 质量；prompt engineering 昂贵且不稳定（CoOp/CoCoOp 需要 learning，牺牲 training-free 特性）。
2. **LG-CLIP 的方案**：用 LLM 生成类相关 prompt → SD 生成合成图像 → 合成图像的特征作为 **visual prototype** → 只保留 visual encoder，摆脱 text encoder 与人工标注。
3. **为什么用 synthetic image prototype**：绕过"文本桥"的不稳定与语义偏差；类标签即可生成精确、无偏、可扩展的 prototypes；推理轻量（LLM/SD 成本离线一次性支付）。
4. **multi-scale 的角色**：论文 §3.3 提出 training-free multi-scale feature extraction（MS）。复杂图像中物体可能只占部分区域，"多尺度 crop 特征聚合"被设计用来捕获视觉焦点、提升鲁棒性；论文 Table 3 消融显示 MS 对各数据集普遍有增益（EUROSAT 最突出）。

---

## B. Exact Pipeline（以代码为准）

### B.1 完整流程（LLM+SD 路径）

```text
class labels（数据集自带）
  → [prompts_gen.py]  xAI Grok API 生成 10 条/类 视觉描述 prompt（coarse→fine 拼接）
  → [llm_sd_gen.py / sd_gen.py]  SD 2.1 生成 Ng=10 张/类（固定全局 seed）
  → [muti_scale_gen_feat.py]  生成图 × 10 scales → CLIP visual encoder → per-sample L2-norm features
      → triangular 加权融合 + L2-norm → gen 多尺度特征 (C, Ngen, D)
      → 每类平均池化（mega.py 内）→ class prototype (C, D)

真实测试图
  → [vanilla_clip_ms.py]  真实图 × 10 scales → CLIP visual encoder → per-sample L2-norm features
      → triangular 加权融合 + L2-norm → real 多尺度特征 (N, D)

分类（mega.py）
  → simi_scores = test_f_ms @ prototypes.T   （内积，非严格 cosine，因 prototype 未再归一化）
  → ŷ = argmax_c simi_scores
```

### B.2 单尺度路径（vanilla_clip.py + gen_feat.py）

- `vanilla_clip.py`：real 图像用 `clip.load()` 返回的默认 `preprocess`（Resize(224, bicubic) + CenterCrop(224)）。
- `gen_feat.py`：生成图同样用默认 `preprocess`；特征 reshape 为 (C, Ngen, D)。
- `mega.py`（默认 `--ms1 "" --ms2 ""`）：相同结论 —— prototypes 由生成图特征平均，argmax 内积。

---

## C. Multi-scale 实现细节（代码级确认）

**以 `vanilla_clip_ms.py` 与 `muti_scale_gen_feat.py` 为准（两者逻辑完全对称）：**

| 项目 | 实际情况（代码） | 代码位置 |
|---|---|---|
| scale 数量 | 10 | `for scale in range(1, 11)`（L197 / L231） |
| scale 定义 | `scale_factor = scale / 10.0` → {0.1, 0.2, ..., 1.0} | 同上 |
| crop 方式 | `transforms.RandomResizedCrop(size=224, scale=(sf, sf), interpolation=BICUBIC)` | `vanilla_clip_ms.py` L203-215 |
| crop 尺寸 | min=max=sf，即**面积比固定**；位置与长宽比（ratio 默认 3/4~4/3）随机 | torchvision 语义 |
| resize | 每个 crop 均 resize 到 **224×224** | `size=224` |
| 归一化 | CLIP 官方 mean/std；单样本特征提取后 **L2 normalize** | `get_visualEmbedding` |
| 每 scale 是否独立跑 encoder | **是**。每样本每 scale 一次 visual encoder forward | for 循环内逐个样本前向 |
| fusion 层级 | **feature 级**（非 score 级） | `combine_multi_scale_features` |
| scale 权重 | `w_s = s / 55.0`，55 = 1+2+...+10 | L108 `ratio = scale / 55.0` |
| 权重方向 | scale=10（全图）权重 10/55 最大；scale=1（最小 crop 0.1）权重 1/55 最小 | 同上 |
| 融合后处理 | 加权求和后 **再次 L2 normalize** | `features_all /= norm` |
| real vs gen 对称性 | real 与 gen 使用**相同**的 10-scale、相同权重、相同后处理 | 两文件结构对称 |
| 缓存格式 | real: `multi_scale/CLIP_{bk}_feature_scale{s}.hdf5` keys: `test_f/test_l/all_embeddings`；gen: `multi_scale_gen/{LLM}CLIP_{bk}_feature_gen{N}_scale{s}.hdf5` keys: `gen_f/gen_l` | `load_scale_feature` |
| merge 读取 | `muti_scale_merge.py` 相同权重逻辑；`mega.py` 读取融合后的 `*_feature_ms.hdf5` | 两文件 |
| 单尺度 vs scale10 | **不同**：单尺度=Resize+CenterCrop（保长宽比）；ms 的 scale10=RandomResizedCrop(scale=1.0)（长宽比随机） | 两文件对比 |

### C.1 论文公式 vs 代码的重要差异（已核实为"编号方向"差异）

- 论文 Eq.(2)：$W_n = \frac{N+1-n}{\sum_{i=1}^N i}$，配合 Eq.(3) $CR_n = 1/n$ —— n=1 是全图（CR=1），权重 $10/55$ 最大；n=10 是最小 crop（CR=0.1），权重 $1/55$ 最小。
- 代码：scale=1 → factor=0.1（最小 crop），w=1/55；scale=10 → factor=1.0（全图），w=10/55。
- **结论：论文与代码物理含义一致**（全图权重最大，最小 crop 权重最小），只是编号方向相反。文档与后续实现统一采用**代码编号**（scale s∈{1..10}，factor=s/10，w_s=s/55）。

### C.2 随机性提示（重要）

- `RandomResizedCrop` 每个样本每次调用都会随机采样 crop 位置与 ratio；全局 seed=2024 固定。
- 完整顺序执行（scale=1→10、按 dataset 顺序）时可复现与官方相同的 crop 序列。
- **若跳过/重排 scale，后续 scale 的 crop 会因 RNG 状态推进不同而变化** → instrumentation 与 early-exit 实验时必须验证等价性并采用固定 crop 策略（预采样 crop 参数），详见后续 Phase 4 实现说明。

---

## D. 与复现相关的关键事实

### D.1 数据集约定（`utils/myDataset.py`）

| 数据集 | 期望路径 | split 文件格式 |
|---|---|---|
| PET | `PET/OxfordPets/images/` | `split_OxfordPets.json`：`{"train": [[rel, label, class_name],...], "test": [...]}` |
| EUROSAT | `EUROSAT/2750/` | `split_EuroSAT.json`：`{"train": [[rel, label, class_name],...], "test": [...]}` |
| FLO | `FLO/Flowers102/jpg/` | `split_OxfordFlowers.json`：`{"train": [[rel, label],...], "test": [...]}` + `cat_to_name.json` |
| CUB | `CUB/CUB_200_2011/` | 标准文件（image_class_labels.txt 等） |
| FOOD | `FOOD/images/` + `FOOD/meta/classes.txt` | `split_Food101.json` |
| ImageNet | `ImageNet/images/ILSVRC2012_img_val/<synset>/*.JPEG` | 无（目录即标签） |

**关键风险：`split_*.json` 不在 repo 中**，需自行构建。论文 §4.1 明确："Since the training process is not involved, we use all the original images from each dataset as the test set."
→ 构建策略：**test split = 数据集全部图像**；train split = 官方给定（若存在）或占位小子集（不影响 test 评估；`vanilla_clip.py` 会冗余提取 train 特征但 `mega.py` 不使用）。

### D.2 论文报告数字（复现参照，Table 1 "Ours" = LLM+SD+MS；Table 3 消融）

| Dataset | Backbone | CLIP (single, text proto) | CLIP&LLM | CLIP&MS | CLIP&LLM&MS |
|---|---|---|---|---|---|
| EUROSAT | ViT-B/32 | 22.22 | 27.84 | 31.91 | **32.42** |
| EUROSAT | ViT-B/16 | 18.31 | 40.20 | 36.02 | **42.43** |
| EUROSAT | ViT-L/14 | 27.78 | 47.74 | 53.04 | **56.20** |
| EUROSAT | RN50 | 11.84 | 27.16 | 18.74 | **34.67** |
| ImageNet | ViT-B/32 | 44.29 | 42.17 | 46.32 | **46.72** |
| ImageNet | ViT-B/16 | 49.56 | 50.03 | 52.27 | 51.89 |
| ImageNet | ViT-L/14 | 56.49 | 57.31 | 58.56 | **59.67** |

Table 1 其余数据集（"Ours" 列，即 LLM+MS）：
- CUB: RN50 34.55 / B32 44.98 / B16 49.57 / L14 57.47
- FLO: RN50 33.58 / B32 45.96 / B16 50.06 / L14 61.35
- PET: RN50 52.61 / B32 63.17 / B16 71.13 / L14 78.98
- FOOD: RN50 57.05 / B32 63.75 / B16 67.18 / L14 77.43

> 注意：Table 1 中 "CLIP" 列为标准 CLIP 文本原型单尺度 baseline；"Ours" 为 LG-CLIP（生成图原型+LLM+MS）。

### D.3 生成阶段参数（`sd_gen.py`）

- SD 2.1（`--sd_version 2.1`），每类 `Ngen=10` 张，固定 seed=2024；生成目录 `SD_gen/SD_2.1_{dataset}_10/{class_name}/`。
- LLM 路径需要 xAI Grok API key（`prompts_gen.py --api_key`）。**本机无该 key；本轮复现走 plain-SD 路径为主（class label prompts），LLM 路径标记为不可用（记录在案）。**

---

## E. 对 AdaScale 研究最重要的 5 个事实

1. **multi-scale = 10 次独立 visual encoder forward/样本**，每个 scale 面积比固定、位置/长宽比随机，输出 224×224。
2. **融合在 feature 级、权重 triangular（w_s = s/55）、融合后 L2-normalize**；prefix 分析时权重可只取相对比例 $w_s = s$（L2-normalize 消除标量因子）。
3. 真实图与生成图**完全对称**地做 multi-scale；prototype 由融合后的生成图特征平均（未再 normalize）。
4. **单一 scale 的独立准确率在官方脚本中已逐 scale 打印**，但官方未研究"累计 prefix"预测动态，也未研究 early-exit——这正是本项目的切入点。
5. 官方评估是**每 scale 独立 acc 报告**；没有保存 per-sample 每 scale 预测 → 需要 instrumented 重跑（保持数值一致）。

---

## F. 复现风险清单（与缓解措施）

| 风险 | 级别 | 缓解 |
|---|---|---|
| split json 缺失 | 高 | 自建构建脚本；test=全部图像（与论文声明一致）；记录每个数据集实际样本数 |
| LLM (Grok) prompt 无法获取 | 中 | 先走 plain SD；论文中 MS 的增益独立于 LLM 即成立（Table 3: CLIP&MS 行） |
| SD 2.1 生成图与作者不完全一致 | 中 | 固定 seed（官方做法）；承认生成的类内多样性差异，本研究的核心（multi-scale 冗余）对生成集来源不敏感 |
| RandomResizedCrop RNG 一致性 | 高 | instrumentation 时验证 full-10 acc 完全一致；方法阶段采用预采样 crop |
| CLIP 权重下载到项目外 | 低 | HOME/cache 已重定向至项目内 `.home`/`.cache`，需验证 |
