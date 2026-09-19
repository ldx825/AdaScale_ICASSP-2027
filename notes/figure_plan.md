# Figure Plan — 全图清单与论文位置规划

日期：2026-09-11（深挖阶段后）。
状态：**主文图已全部就位并编译通过（4 页正文 + 第 5 页参考文献）**；其余图为补充材料/备用。

## 论文主文（当前编译状态）

| # | 文件 | 内容 | 位置 | 状态 |
|---|------|------|------|------|
| Fig 1 | `paper/figures/architecture/fig_arch.pdf` | **总体架构图**：Level-1 无标签熵诊断（ΔH>θ?）→ SINGLE / Level-2 尺度循环可靠性停止 → 概率软融合输出 | §3 Method 开头，单栏 | ✅ 在文 |
| Fig 2 | `paper/figures/figA_pet_text.pdf` + `figA_eurosat_text.pdf` | 两 regime 前缀曲线（Pets 融合赢 / EuroSAT 融合输） | §4.2，单栏双拼 | ✅ 在文 |
| Fig 3 | `paper/figures/deep_diag/fig_attn_eurosat.pdf` | 单案例跨尺度 attention：s=1,3 自信错 → s≥5 恢复 | §4.2 后，跨栏 | ✅ 在文 |
| Fig 4 | `paper/figures/deep_diag/figM_compact.pdf` | 机制双联：四象限 + NQ 救回率（Pets/Flowers 92–97% vs EuroSAT 76–85%） | §4.4 后，单栏 | ✅ 在文 |
| Table 1 | `paper/tables/tab_audit.tex` | 尺度效用审计（含 CUB 行） | §4.2 | ✅ 在文 |
| Table 2 | `paper/tables/tab_baselines.tex` | prior-inspired 基线对照（含 CUB 行） | §4.4 | ✅ 在文 |
| Table 3 | `paper/tables/tab_main.tex` | 主表（7 设置，v3 数字） | §4.3 | ✅ 在文 |

注：排版用 `\enlargethispage{2.5\baselineskip}` 收口为 4 页正文 + refs 页。

## 补充材料（已画好，文件就绪）

**正式补充材料文档**：`paper/supplement.tex` → `supplement.pdf`（4 页，0 error）。
内容：S1 实现细节（提取协议/CUB/冻结超参）、S2 Algorithm 1 伪代码、
S3 选择器灵敏度（子采样/θ 容差/跨骨干/安全失败模式）、S4 附加诊断（轨迹图 S1、
四象限+drag S2、attention 定量负结果、几何图 S3）、S5 附加结果（B/16 全表、
runtime 表、v3 灵敏度与增益分解、NQ 机制）。

| # | 文件 | 内容 | 说明 |
|---|------|------|------|
| Fig S1 | `figures/deep_diag/figT_trajectory.pdf` | prefix margin/熵轨迹（early vs full-correct vs full-wrong，4 FUSED 设置） | 支持早停机制；"早退样本快速稳定、走满错误样本熵最高" |
| Fig S2 | `figures/deep_diag/figG_geometry.pdf` | cos(f_s, gt) 与几何 margin 随尺度 | 几何视角；gen 原型 margin 平台为负等现象 |
| Fig S3 | `figures/deep_diag/figM_mechanism.pdf` | Fig 3 的三联大版（含 label-free 拖拽曲线 (c)） | 拖拽曲线：EUROSAT/gen 被小尺度拖拽 18.4%（其余 ≤6.8%） |
| Fig S4 | `paper/figures/figB_pareto.pdf` | accuracy–compute Pareto 6 子图 | 全部方法（含 Fixed-K、B1、B2、B3） |
| Fig S5 | `figures/attention/EUROSAT_ViT-B32_text_cases_grid.png` | EUROSAT 3 案例 × 4 尺度网格 | 多案例版 attention |
| Fig S6 | `figures/attention/PET_ViT-B32_text_cases_grid.png` | PET 2 案例 × 4 尺度网格 | PET 融合救回案例 |
| Fig S7 | `figures/deep_diag/fig_attn_pet.pdf` | PET 单案例 attention（s=1 局部误判 → s≥3 恢复） | 与 Fig 2 对应的 PET 版 |
| Table S1 | `paper/tables/tab_runtime.tex` | 方法级 wall-clock（PET 290/127/189/202 s 等） | 主文只留 1.44×/8.8× 数字，表移补充 |
| Table S2 | `outputs/final/all_results_v2.csv` 全列 | v2/v3、fixed5/7/9、p 值、rel-entropy 差 | 数据表 |

## 备用/未用图的说明（诚实记录）

1. **attention 定量统计**（`outputs/attn_stats/EUROSAT_ViT-B32_text.json`）：CLIP 注意力在全部 12 层都极度均匀（归一化熵 > 0.91；rollout 后 0.998 恒定），**逐样本/逐尺度无区分度**——该负结果不支持其作为论文图表；Fig 2 的 overlay 已注明"per-crop min–max normalized"。Fig 2 的价值在定性机制展示。
2. **v3 增益分解**（早退组 Δ=+0.00，走满组 +0.27~0.56）：数字写入 `notes/v3_design.md`，暂未做图（如需要可 10 分钟产出条形图）。

## 生成脚本索引

| 图 | 脚本 | 命令 |
|----|------|------|
| Fig 2 / S5 / S6 / S7 | `scripts/analysis/fig_attn_case.py`、`scripts/analysis/attention_maps.py` | `--dataset EUROSAT --proto text --case_idx 15057 --out fig_attn_eurosat` |
| Fig 3 / S1 / S2 / S3 | `scripts/analysis/plots_deep_diag.py`（读 `outputs/deep_diag/*.json`） | `python scripts/analysis/plots_deep_diag.py` |
| S4 | `scripts/paper/fig_pareto.py` | 直接运行 |
| 数据 | `scripts/analysis/deep_diag.py` | `--dataset X --proto Y`（6 设置） |
| 注意力统计 | `scripts/analysis/attn_stats.py` | `--dataset EUROSAT --proto text --n 600` |

## 复现编译

```bash
cd AdaScale-CLIP/paper
latexmk -pdf -interaction=nonstopmode main.tex   # 4pp 正文 + refs 页, 0 error
```
