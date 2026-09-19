# LG-CLIP Code Map（instrumentation 定位）

> 目标：明确各文件职责、数据流向，以及**在哪里插入 per-scale logging** 才能在保证 baseline 预测数值完全不变的前提下，dump 每个 scale 的 logits / prediction / feature。

---

## 1. 模块职责图

```text
vanilla_clip.py            real 图单尺度特征（Resize+CenterCrop）→ CLIP_{bk}_feature.hdf5
    └── get_textEmbedding  文本原型（"A photo of a {cls}."）→ all_embeddings
    └── get_visualEmbedding 逐样本 L2-norm visual feature

vanilla_clip_ms.py         real 图多尺度特征（10 scales，RandomResizedCrop(scale=sf)）
    ├── 每 scale → multi_scale/CLIP_{bk}_feature_scale{s}.hdf5 (test_f, test_l, all_embeddings)
    ├── 每 scale 独立 acc 打印（sanity check）
    └── combine_multi_scale_features → CLIP_{bk}_feature_ms.hdf5（triangular w=s/55 → L2-norm）

sd_gen.py                  class label → SD 2.1 生成图（SD_gen/SD_2.1_{ds}_10/{cls}/）
llm_sd_gen.py              LLM prompt 版生成（LLM_SD_gen/LLM_SD_2.1_{ds}_10/{cls}/）
prompts_gen.py             Grok API 生成类描述 prompt（需 API key；本机不可用）

gen_feat.py                生成图单尺度特征 → {LLM}CLIP_{bk}_feature_gen{N}.hdf5 (gen_f: C×Ngen×D)
muti_scale_gen_feat.py     生成图多尺度特征（与 vanilla_clip_ms 对称）
    └── multi_scale_gen/{LLM}CLIP_{bk}_feature_gen{N}_scale{s}.hdf5
    └── → {LLM}CLIP_{bk}_feature_gen{N}_ms.hdf5

muti_scale_merge.py        独立重融合工具（real / gen 两模式）
text_gen_Ngen_made.py      Ngen≠10 时的图像筛选（top-N by CLIP sim）

mega.py                    最终评估：gen prototype 平均 → 内积 → argmax → acc → metalog.txt
```

## 2. 数据流（multi-scale 完整版）

```text
real images ──► vanilla_clip_ms.py ──► multi_scale/…_scale{1..10}.hdf5 ──► merge(w=s/55, L2norm)
                                                                              └──► …_feature_ms.hdf5 ─┐
                                                                                                     ▼
gen images ──► muti_scale_gen_feat.py ──► multi_scale_gen/…_scale{1..10}.hdf5 ─┘           mega.py: sim = test_f @ mean(gen_f).T
```

## 3. Instrumentation 插入点分析

### 3.1 约束
1. 不得改变 baseline prediction（full-10 的融合特征与预测必须与官方脚本**逐位一致**或误差 ~0）。
2. 需要 dump 的信息：
   - 每样本每 scale 的 **feature**（L2-norm 后的向量，或至少其与 prototypes 的相似度）。
   - 每样本每 scale 的 **top-1 / top-2 / margin**（基于 prefix 聚合后特征的预测）。
   - 验证用：scale 独立预测（脚本已打印单 scale acc）。

### 3.2 方案对比

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A. Fork 官方脚本，加 dump | 复制 `vanilla_clip_ms.py` 为 instrumented 版本，在每 scale 的循环内部 dump `test_f` 每样本特征（同时保持 transform 顺序、seed 不变） | 与官方脚本共享全部逻辑；只加输出不改计算 | 需要严格保持 RNG 调用序列一致 |
| B. 直接读官方缓存 | 官方脚本已保存每 scale 的 `test_f`！仅需额外 dump prototypes | 零计算修改 | 缺少 prototypes（text embeddings 在 hdf5 的 `all_embeddings` 里，有！）；但**per-sample prefix 预测需要知道样本顺序**，hdf5 顺序即 testdf 顺序，OK |
| C. 独立重实现 | 在 `src/analysis/` 中重实现，仅用官方产物对比验证 | 控制力最强 | 必须证明与官方等价 |

**选定方案：A + B 结合。**
- 官方脚本已缓存每 scale 的 per-sample 特征（`multi_scale/…scale{s}.hdf5` 的 `test_f`）+ `all_embeddings`（text prototypes）。
- 因此 **prefix 预测计算完全可以在缓存特征上离线完成**（不需要重跑 encoder）：
  - prefix 聚合：$\tilde f_t = \sum_{s \le t} s \cdot f_s$，$\hat f_t = \tilde f_t / \|\tilde f_t\|$
  - 对 text prototypes（以及后续的 gen prototypes）做内积 → top-1/top-2/margin。
- 唯一需要 instrumented 重跑的场景：需要**样本级随机 crop 的确定性控制**时（early-exit 实验），采用预采样 crop 参数的等价实现。
- 为 Gen 特征（LG-CLIP visual prototype）也要拿到每 scale 的 gen 特征 —— 官方缓存同样有（`multi_scale_gen/…`）。

### 3.3 数值一致性验证协议
对每个数据集：
1. 运行官方 `vanilla_clip_ms.py`，记录每 scale 独立 acc + merge 后特征 `CLIP_{bk}_feature_ms.hdf5`。
2. 在 `src/analysis/` 中从 per-scale 缓存重算 merge（相同权重 s/55 + L2-norm），与官方 `*_feature_ms.hdf5` 逐元素对比（max abs diff < 1e-6）。
3. 用重算特征重跑 `mega.py` 等价计算，acc 与官方 stdout 一致。

> 若验证通过，则所有 prefix/audit 分析均可基于缓存离线完成，避免重复 GPU 计算、避免 RNG 问题。

## 4. 输出目录规划（项目内）

```text
outputs/baseline/<dataset>/<backbone>/     # 官方脚本 stdout 日志、acc 记录
outputs/scale_audit/<dataset>/<backbone>/  # per_sample.parquet、per-scale logits
figures/audit/                             # audit 图表
notes/scale_utility_findings.md            # audit 结论
```

## 5. 关键路径速查

| 内容 | 路径 |
|---|---|
| 官方 repo | `third_party/LG-CLIP`（commit e0a3c6f） |
| 数据根（传给 --image_root） | `$PROJECT_ROOT/data` |
| real 多尺度缓存 | `data/<ds>/multi_scale/` |
| gen 多尺度缓存 | `data/<ds>/multi_scale_gen/` |
| 融合特征 | `data/<ds>/CLIP_{bk}_feature[_ms].hdf5`, `data/<ds>/{LLM}CLIP_{bk}_feature_gen{N}[_ms].hdf5` |
| 生成图 | `data/SD_gen/SD_2.1_<ds>_10/<class>/`（plain） |
