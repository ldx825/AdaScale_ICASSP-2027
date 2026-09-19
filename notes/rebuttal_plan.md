# Rebuttal 弹药包（rebuttal_plan.md）

日期：2026-09（投稿前备好）。用途：审稿意见到达后快速组织回复。
规则：所有数字来自 `outputs/`（可溯源）；引用时保持与论文一致的"安全声明"口径。

---

## Part 1 · 新颖性类质疑

### Q1. "这与 AdapTTA / Selective-TTA 等自适应 TTA 有何本质不同？"

**回应**：它们的自适应对象是"每样本要不要多做几次增强"（per-sample compute allocation），而本文发现的问题是**融合算子本身的域级失效**——我们在论文中用受控实验证明 per-sample 门控无法诊断它：
- B1（margin 停止，AdapTTA 风格）：EuroSAT/gen **27.69%**，比全融合低 8.6 点——"自信地错"的小尺度前缀通过任何置信阈值；
- B2（不确定性门控，Selective-TTA 风格）：EuroSAT 全部阈值 **≤44.93/39.46 < SINGLE 45.34/41.17**，永远追不上；FUSED 域达到同精度需 9.1 视图（AdaScale 6.6~8.9）。
→ 决策层级不同：先"这个域该不该融合"（域级），再"这个样本要几个尺度"（样本级）。

**数据**：Table II（tab_baselines）；`outputs/prior_baselines/*.json`；`notes/baseline_comparison_plan.md`。

### Q2. "TTA 有害不是 ICCV'21 (Better Aggregation) 已经发现了吗？"

**回应**：是的，我们明确引用并**不主张首发**。我们的发现是**不同的现象**：不是个别样本的翻转，而是**整个目标域上融合系统性劣于最大单尺度**（EuroSAT 两种原型协议皆然：42.38<45.34、36.33<41.17），且给出微观机制（NQ 救回率 84.6/76.2 vs 其他域 92~97）与修复方式（域级诊断）。论文 Intro 第 2 段已写明 "Consistent with prior observations..."。

**数据**：Fig. 4（机制）、Table I（audit）；`outputs/deep_diag/*.json` 的 quadrants。

### Q3. "样本级早停不新。"

**回应**：同意，我们将其定位为**支持性机制**（Sec. 3.4 明确 "Building on the general principle of confidence-aware adaptive TTA..."）；论文的三个贡献中不含"早停"本身。早停的作用是在 FUSED 域省算力（10–90%，均值 38%），且被证明不损害精度（7/7 设置无显著负项）。

---

## Part 2 · 方法类质疑

### Q4. "为什么用熵而不是更简单的信号（如拖拽统计/agreement）？"

**回应**：我们做了直接对比（supplement S6）：在 11 个设置上，
- 熵信号 ΔH：**11/11 全对**，SINGLE/FUSED 分离度 **+2.20pp**（无边界情形，最小 margin 0.54pp）；
- 拖拽信号（融合跟随 s1 的比例）：10/11（EuroSAT/text 失败），分离度仅 +1.01pp。
拖拽信号保留在论文中作为**机制解释**（Fig. 2c/supplement），但不作为诊断器。

**数据**：`outputs/deep_diag/signal_compare.json`；`scripts/analysis/diagnosis_signal_compare.py`。

### Q5. "熵诊断会不会是碰巧？换个骨干就失效？"

**回应**：跨骨干复现：B/16 上全部 4 个设置选择正确（PET text/gen、EuroSAT、Flowers），ΔH 值 PET −0.12/−0.65%、EuroSAT +6.15%、Flowers +0.18%（阈值 1.0%）。合计 11 个设置零错误。
且诊断统计**用无标签校准集**（n=500），均匀子采样 200 次决策稳定率 98–100%（n=1000 时 100%），θ∈[0.5%, 2.3%] 内结论不变——容差很宽。

**数据**：`outputs/final/*ViT-B16*.json`；`outputs/mode_selector/*.json`；论文 §4.5。

### Q6. "τ_s=0.02 是不是在测试集上调出来的？"

**回应**：不是。τ 是先验设定（软化=缓解单视图过尖分布），且敏感性已测：τ∈{0.01, 0.02, 0.05} 在全部 6 个（后为 7）融合设置上非负（v3_sim 的 F1/F7 行）；0.02 是设计直觉 + 稳定性选择，非测试集最优搜索。论文 Sec. 3.5 明确 "fixed a priori"。

**数据**：`outputs/final/v3_sim/*.json`（F1/F7/F7τ0.05 行）。

### Q7. "v3（软融合）增益只有 +0.16，值得写吗？"

**回应**：值得，因为它是**零成本无损增益**：决策层完全冻结（平均视图数不变、运行时不变），增益集中在难样本（走满组 +0.27~0.56），早退样本 Δ=0.00（零副作用）。7/7 设置非负。相对之下，B3 类过滤方法的改进需要付全 10 视图的算力。

**数据**：增益分解（`notes/v3_design.md`）；`outputs/final/v3_sim/*.json`。

---

## Part 3 · 实验类质疑

### Q8. "为什么 4 个数据集里 3 个是 FUSED？SINGLE 结论是否只靠一个数据集？"

**回应**：SINGLE 域（域级融合失败）在 EuroSAT 上**两个独立协议**（text/gen 伪标签）+ **两个骨干**（B/32、B/16）共 4 个设置中全部成立（42.38/36.33/40.47(融合) vs 45.34/41.17/42.74）。更重要的是机制量（NQ rescue 76.2–85.1% vs FUSED 域 91.9–97.5%）表明这是**可测的域性质**，非单点巧合。CUB-200（后加的第 4 数据集）是 FUSED——我们**如实报告**而非挑选。若审稿人要求更多 SINGLE 域，我们已扫描的候选（DTD/Resisc45）可作为 future work——**注意**：不要在反驳中承诺未跑的实验。

**数据**：Table I/III；`outputs/deep_diag`。

### Q9. "效率数字是不是只看尺度数？"

**回应**：不是，附真实 wall-clock（同机同 crop，RTX 4080 Super）：PET 202s vs 290s（1.44×，同精度）；EuroSAT 45s vs 397s（8.8×，且精度 +3.0）。supplement Table S1 有完整表。

### Q10. "有没有跟 TPT/DiffTPT 直接比较？"

**回应**：协议不同（他们需要 prompt 优化/扩散生成，属训练式适配），不在同一 setting；我们**借鉴其视图过滤思想**构造了 B3 熵过滤基线并全面比较（被 AdaScale Pareto 占优：B3 需全 10 视图且修不了 EuroSAT 43.42%<45.34%）。相关工作的区分写在 Sec. 2。

### Q11. "B1 崩溃（27.69%）是不是超参没调好？"

**回应**：δ 全扫描 {0.2, 0.35, 0.5, 0.65, 0.8}，最优仍是 27.69–42.39% 区间（对应不同视图数），全部显著低于 SINGLE 45.34%（EuroSAT/gen 全部 < 41.17%）。原因是结构性的：margin 阈值再调也无法滤掉**置信度不低的错误**（如 attention 案例显示 s=1 预测 Highway p=0.51）。

**数据**：`outputs/prior_baselines/EUROSAT_ViT-B32_gen.json`（margin_stop 全扫描）。

---

## Part 4 · 快速数据索引（一行一条）

| 需求 | 文件 |
|---|---|
| 主表数字 | `outputs/final/all_results_v2.csv` |
| B1/B2/B3/WGT/SNAP/LOO | `outputs/prior_baselines/*.json` |
| v3 候选扫描 | `outputs/final/v3_sim/*.json` |
| 四象限/拖拽/轨迹 | `outputs/deep_diag/*.json` |
| 信号对比 11 设置 | `outputs/deep_diag/signal_compare.json` |
| B/16 评估 | `outputs/final/adascale_v2_*ViT-B16*.json` + `v3_sim/FLO_ViT-B16_text.json` |
| 配对 bootstrap | `outputs/stats/*_paired.csv` |
| 运行时 | `outputs/efficiency/` + `paper/tables/tab_runtime.tex` |
| attention 案例 | `figures/attention/`、`figures/deep_diag/fig_attn_*` |
| 审计表 | `outputs/scale_audit/*/ViT-B32/summary_*.json` |
| 新颖性问答 | `notes/novelty_reaudit.md`、`notes/claim_audit.md` |
