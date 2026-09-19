# Novelty Re-Audit（novelty_reaudit.md）

日期：2026-02。用途：逐条评估论文中的潜在 novelty 主张，对照最接近的 prior work。

| Idea / Claim | Closest Prior Work | Overlap | Difference | Can Claim as Novel? |
|---|---|---|---|---|
| Dynamic # TTA views | AdapTTA 2021 (VLSI-SoC) | High | 我们的视图 = 尺度前缀（顺序、可累积融合）；AdapTTA = 增强迭代（无前缀融合结构） | **No**（样本级动态视图数不新） |
| TTA only if uncertain | Selective-TTA 2023 (Inf. Sci.) | High | 我们是**域级**模式切换（融合开/关对整个数据集），它们=样本级门控 | No（样本级门控不新；域级诊断是新点） |
| Two-stage adaptive TTA | Intelligent Multi-View TTA 2024 (ICIP) | Medium/High | 它们的决策 = 增强选择 + 是否 TTA（样本级）；我们的决策 = 域级融合效用 + 样本级尺度预算 | Not broadly（两阶段框架不新） |
| TTA can hurt predictions | Better Aggregation ICCV 2021 | High | 它们关注聚合器改进；我们报告的是 synthetic-prototype 多尺度融合中的**域级系统性失败**（融合 < 最大单尺度，跨 2 种原型） | Only the specific phenomenon |
| CLIP augmented-view confidence selection | TPT / DiffTPT | Medium | 它们用 entropy/confidence 选视图并**更新 prompt**；我们不更新任何参数，且用熵做**域级融合诊断** | Not generic（视图筛选不新） |
| Target-domain SINGLE/FUSED entropy diagnosis | 无直接先例（closest：Selective-TTA 的样本级不确定性门控） | Low | 决策对象是"融合算子在该域是否有益"（域级、label-free、单标量判据） | **Yes — primary novelty** |
| Hierarchical domain+sample controller | 组合（AdapTTA + Selective-TTA 类组件的层级化） | Medium | 先域级模式、再样本级预算；实证显示 per-sample 方法无法替代域级诊断（B1 崩溃 27.69、B2 ≤ AdaScale） | **Yes — secondary novelty**（组合 + 必要性证据） |

## 重要自查结论

1. **"first adaptive TTA" 类表述全部禁用**——AdapTTA 2021 / Selective-TTA 2023 早已做样本级自适应 TTA。
2. **"TTA can harm" 不加"first"**——ICCV 2021 已有；我们只主张"在 synthetic-prototype 多尺度融合中发现域级系统性失败现象"。
3. **最强卖点**：域级融合效用诊断（label-free），且**有对照实验证明** per-sample 门控（B2）无法取代它：
   - EUROSAT：B2 全部阈值 ≤ 44.93/39.46 < SINGLE 45.34/41.17
   - B1（AdapTTA 风格）在 EUROSAT/gen 崩溃至 27.69（比 full-10 低 8.6pt）
4. **第二卖点**：层级控制器的 Pareto 优势（同精度下视图更少；同视图下精度更高），见 baseline_comparison_plan.md。
5. 若后续发现更接近域级诊断的先前工作 → 必须更新本表并修订论文（spec §24 规则）。
