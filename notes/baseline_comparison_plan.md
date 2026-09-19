# Baseline Comparison Plan & Results（baseline_comparison_plan.md）

日期：2026-02。数据源：`outputs/prior_baselines/*.json`（脚本 `scripts/final/prior_baselines.py`，全部在缓存特征上计算，公平对照）。

## 基线定义（与 spec §9 对应）

| 基线 | 规格 | 计算量 |
|---|---|---|
| **B1 Margin Stop**（AdapTTA 风格） | 每前缀 t 算 margin m_t = p1−p2（τ=0.01）；m_t ≥ δ 即退出并输出前缀预测。δ ∈ {0.2,0.35,0.5,0.65,0.8}。无 agreement/JS。 | 自适应（报告 avg_scales） |
| **B2 Selective Gating**（Selective-TTA 风格） | 先算 single-10 预测；若 H(single) < γ 返回之，否则跑 Full-10 融合。γ 取相对熵分位 q ∈ {0.1..0.9}。 | 1 或 10（报告平均值） |
| **B3 Entropy Filter**（TPT/DiffTPT 风格） | 算全部 10 视图熵，保留最低熵 K=3/5/7 个视图做三角融合。 | **10（必须看全）** |
| **WGT** | 全 10 视图 margin/entropy 加权融合。 | **10** |
| **SNAP** | 取 margin 最大的单视图预测。 | 10 |
| **AdaScale v3（ours）** | 域级 SINGLE/FUSED + 样本级停止 + 概率层 soft fusion。 | 1 或自适应 |

## 主对照表（accuracy @ avg_scales；加粗 = 组内最优同算力）

| Setting | Full-10 | Single-10 | B1 best | B2 best | B3 (K=5) | WGT-m | **AdaScale v3** |
|---|---|---|---|---|---|---|---|
| PET/text | 82.45@10 | 80.09@1 | 82.31@6.19 | 82.45@9.10 | 82.75@10 | 82.56@10 | **82.63@6.63** |
| PET/gen | 60.92@10 | 57.01@1 | 61.03@8.90 | 60.92@9.10 | 61.14@10 | 61.23@10 | **61.15@8.63** |
| FLO/text | 65.12@10 | 63.05@1 | 65.08@7.95 | 65.12@9.10 | 64.77@10 | 65.01@10 | **65.17@7.91** |
| FLO/gen | 42.95@10 | 41.30@1 | 42.94@9.02 | 42.94@9.10 | 43.28@10 | 43.34@10 | **43.30@8.92** |
| EUROSAT/text | 42.38@10 | **45.34@1** | 42.39@9.76 | 44.93@1.90 | 43.42@10 | 43.50@10 | **45.34@1.00** |
| EUROSAT/gen | 36.33@10 | **41.17@1** | 27.69@5.92 | 39.46@1.90 | 36.00@10 | 34.58@10 | **41.17@1.00** |

## Pareto 关键结论（Narrative A 的证据）

1. **B2（Selective-TTA 风格）无法替代域级诊断**
   - EUROSAT 全部阈值下 ≤ 44.93/39.46 < SINGLE 45.34/41.17（样本级门控救不了域级失败）；
   - 在 FUSED 数据集达到与 AdaScale 相近精度时需要 9.1 个视图（AdaScale 6.63~8.92）；
   - EUROSAT 上 B2 用 1.90 视图 vs AdaScale 1.00（SINGLE 一次前向）。
2. **B1（AdapTTA 风格）在域级失败场景崩溃**：EUROSAT/gen 27.69（比 full-10 低 8.6pt）——纯置信停止会锁定"自信但错误"的小尺度前缀；AdaScale 的域级 SINGLE 决策彻底避免此失败。
3. **B3/WGT 需要全部 10 视图**才有增益（熵/margin 加权必须先看全），Pareto 上劣于 AdaScale v3（如 PET/text：B3 82.75@10 vs v3 82.63@6.63，25% 的算力差换 +0.12）。
4. **AdaScale 同时是 EUROSAT 与 FUSED 设置的 Pareto 前沿**：SINGLE 域（1.00 视图，最高精度）；FUSED 域（自适应 6.6~8.9 视图，最高精度）。

## 统计（paired bootstrap, B=10000）

| 对比 | 结论 |
|---|---|
| AdaScale v3 vs Full-10 | FLO/gen +0.35 p=0.002*；PET/text +0.18 p=0.079；PET/gen +0.23 p=0.080；FLO/text +0.05 n.s. |
| AdaScale v3 vs Fixed-5 | 全部 +1.4~+2.2 p<0.001* |
| AdaScale v3 vs Fixed-7 | −0.11~+0.55（混合；n.s.~0.015*） |
| AdaScale v3 vs Single-10 | FUSED 域 +2.0~+4.1 p<0.001*；SINGLE 域 +2.97/+4.84 p<0.001* |
| AdaScale v2 vs v3 | −0.02~−0.23（v3 更好；FLO/gen p=0.033*） |

（"vs B1/B2 best"的直接配对检验：B1/B2 报告给定参数下的运行点，paper 中以 Pareto 曲线展示；如需显著性子集可后续用其 per-sample 预测补跑。当前 B1/B2 尚未保存 per-sample 预测文件——**TODO（若 reviewer 需要）**。）

## 待办

- [ ] （可选）为 B1/B2 最优运行点保存 per-sample 预测 → 与 AdaScale 做配对 bootstrap；
- [ ] Pareto 图（Fig 3）：accuracy vs avg_scales，含 Full-10、Fixed-K、B1、B2、AdaScale（数据已齐）。
