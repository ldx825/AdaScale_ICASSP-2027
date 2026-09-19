# Paper Repositioning Changes（paper_repositioning_changes.md）

日期：2026-02。依据 spec §27 决策：**选择 Narrative A**。

## Narrative 决策

> **Narrative A — Hierarchical target-domain fusion diagnosis is necessary beyond standard per-sample adaptive TTA.**

证据（详见 baseline_comparison_plan.md）：
- B2（Selective-TTA 风格样本级门控）在 EUROSAT 全部阈值 ≤ 44.93/39.46 < 45.34/41.17 → 无法替代域级诊断；
- B1（AdapTTA 风格 margin stop）在 EUROSAT/gen 崩溃至 27.69；
- B3/WGT 需要全部 10 视图才有效，Pareto 劣于 AdaScale v3；
- AdaScale 在 SINGLE 与 FUSED 两域均为 Pareto 前沿。

## 变更清单（main.tex）

### 标题
- 旧：`AdaScale: Training-Free Adaptive Multi-Scale Inference for Synthetic-Prototype Zero-Shot Recognition`
- 新（候选 1，选用）：**`AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition`**
- （备选 3：`Not Every Domain Needs Fusion: ...`；备选 2：`When Should Scales Be Fused? ...`）

### Abstract（第 6 节结构）
1. 合成原型零样本管线使用固定多尺度融合；
2. 固定融合假设"所有域都受益于全尺度组合"；
3. 审计发现 ①晚期尺度会翻转个体正确预测（Regime A，Pets）；②整个域上融合系统性劣于最大单尺度（Regime B，EuroSAT；文本与合成原型皆然）；
4. AdaScale 用**无标签域级熵统计**先决定 SINGLE vs FUSED；
5. 若 FUSED，按样本可靠性停止（agreement+margin+JS drift），输出层用概率软融合；
6. 只报告已验证数字；禁用 first/never-studied/新发现伤害等表述。

### Introduction（5 段逻辑，第 5 节）
1. CLIP 零样本 + 合成原型 + 多尺度聚合 + 计算代价（无 novelty 声明）；
2. 承认 prior adaptive TTA（AdapTTA/Selective-TTA/Multi-View TTA）与 harmful aggregation（ICCV'21）；转折句点名"域级融合失败"；
3. 审计两 regime（Pets / EuroSAT）→ "fusion utility is a target-domain property, whereas scale sufficiency is sample-dependent"；
4. AdaScale 简介（L1 → L2）；
5. 三条贡献（审计 / label-free 域级诊断+层级控制器 / 实验；早退不进主贡献）。

### Related Work（三小节，第 7 节）
- 7.1 合成原型零样本识别（LG-CLIP + 审计此管线）
- 7.2 自适应 TTA（AdapTTA/Selective-TTA/Multi-View TTA/RL 2026；"prior work already establishes that TTA compute can be allocated adaptively"）
- 7.3 聚合与视图可靠性（Better Aggregation/TPT/DiffTPT；"entropy/confidence/filtering are prior art"）

### Method（重排，第 8 节）
- 3.1 多尺度合成原型推断（baseline 定义）
- 3.2 尺度效用审计/问题形式化（fusion utility vs scale sufficiency；标签仅用于离线分析）
- 3.3 **无标签融合效用诊断**（熵差方程；proxy 措辞）
- 3.4 尺度级可靠性停止（cite adaptive-TTA；支持性定位）
- 3.5 层级推断算法（伪代码）；**新增 3.6 概率层软融合（v3）**

### Experiments
- 新增 **Table II：Strong baselines / ablation**（Full-10 / Fixed-K / Margin Stop / Selective-TTA-style / Entropy Filter / AdaScale w/o L1 / AdaScale full）
- 主表更新为 AdaScale v3 数字（已完成，tab_main.tex）
- 模式选择行为（6/6 正确）+ 敏感性
- Runtime：Full-10 / Single / Fixed-K / Selective-TTA / AdaScale（已有 PET 数据）
- 定性：attention 案例图（小尺度自信错 → 大尺度对；PET 融合救回）

### Conclusion
- 中心 = 融合诊断；早退次要；删除广域"adaptive scale"措辞。

## 完成状态

- [x] 7 篇 prior work 验证（related_work_matrix.md）
- [x] claim_audit.md
- [x] novelty_reaudit.md
- [x] baseline_comparison_plan.md（含结果）
- [x] adascale.bib 增补（7 条）
- [x] main.tex 重写（abstract/intro/related/method/experiments/conclusion）
- [x] 主表 v3 数字
- [ ] Fig 2（两 regime 机制）/ Fig 3（Pareto）——绘制中
- [ ] 长度检查（ICASSP 4 页）
