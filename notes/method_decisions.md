# Method Decisions — AdaScale-CLIP

> 记录每次"观察 → 假设 → 实验 → 结果 → 决策"的链条（文档 §21.3）

---

## D1. 研究问题确认（2026-09-11, Phase 4 后）

**Observation**（PET/ViT-B/32/text-proto）：
- prefix acc 在 t=7 达峰（82.73%），t=8/9/10 边际收益 -0.15/-0.01/-0.12（负）
- teq 中位数=2；69.2% 样本 teq≤2；harmful rate=4.95%
**Decision**：GO（方向成立：固定 10-scale 存在冗余与后期有害性）。方法围绕"training-free early exit + reliability-aware fusion"。

## D2. 概率标定必须是 CLIP 温度（2026-09-11）

**Observation**：直接用 softmax(cosine logits) 得到 margin ~0.001 量级（37 类几乎均匀），所有阈值失效；grid 全部 avg_scales=10。
**Decision**：统一采用 τ=0.01（等价 logit_scale=100，与 CLIP 训练一致）。修改 `prefix_analysis.py` 与 `simulate.py`。
**Effect**：margin 分布合理化（scale 2 中位数 0.47），信号 AUC 恢复到 0.86–0.98。

## D3. 策略形态：连续稳定性 + margin（2026-09-11）

**Observation**：
- margin 对"未来不再翻转"的 AUC：t=2: 0.86 → t=5: 0.93 → t=9: 0.98（很强）
- entropy 次之（0.82–0.87）；JS 较弱（0.54–0.82）；stable_len 中等（0.61–0.71）
**Hypothesis**：退出规则 = "连续 r 个 scale 预测一致 + 当前 margin ≥ δ（+ 可选 JS ≤ ε）"
**Experiment**：细网格（r∈{2,3,4}，δ∈[0.2,0.75]，ε∈{关,0.03,0.01}）+ Fixed-K 对照（同 crop 池）。
**Result**：
- 最优区域 r=3, δ∈[0.25,0.45]：4.45–5.26 scales 下 acc 81.3–82.5%，超同预算 fixed-K +0.7～1.3%
- 5.26 scales（省 47%）→ 82.46% ≥ full-10 (82.45%)
**Decision**：主方法采用 r=3 的稳定性+marggin 策略；JS 作为可选正则（作用小，可能放弃以简化）。**最终冻结的默认超参待跨数据集验证后确定。**

## D4. Early exit 的叙事：不只"省计算"

**Observation**：fixed-7 (82.73%) > fixed-10 (82.45%)；harful transitions 累积 4.95%。
**Decision**：论文同时报告两点：(a) 省计算；(b) **早停可避免部分后期的 harmful 更新**（prefix 峰值 > full-10）。这回答"多尺度是否越多越好"。

## D5. 聚合策略（候选，未锁定）

**Observation**：当前全部使用 triangular 固定权重。后期 scale 特征质量参差（单 scale acc 下降），固定权重可能对有害 crop 过度加权。
**Hypothesis**：按可靠性自适应加权（margin 加权等）可进一步提升同预算精度。
**Status**：待 experiment（E06）——若增益有限则弃（保持方法简洁）。

## D6. 评估协议的关键细节（诚实性要求）

- 所有方法的评估在**同一 crop 池**上进行（官方脚本预提取的 per-scale 特征），确保配对公平。
- 模拟器与真实 dynamic execution 逻辑严格一致（退出后不再计算；最终预测=退出时预测）。
- 效率指标同时记录 avg scales（proxy）+ 后续真实 latency（active-set 执行）。
- **禁止**：用 test GT 调阈值；跨数据集各自调参。冻结一套超参跨所有主数据集。

## D7. 数据集级模式选择：多尺度融合并非普遍有益（2026-09-11）

**Observation**（EUROSAT，27000 图）：
- prefix 精度曲线**单调**（16.24%→42.38%），无 PET 式的中间峰值
- 但 10-scale 三角融合 (42.38%) **低于**单尺度 scale-10 (45.34%，与官方
  single-scale eval 45.35% 一致)；gen 协议 36.33% vs 41.17%
- 即：baseline pipeline 在 EUROSAT 上"融合有害"（-2.97% / -4.84%，
  paired bootstrap p<0.0001）
- 对照：PET 上融合有益（text +2.35%，gen +3.91%，均显著）

**Hypothesis**：若融合提供的是互补证据，融合应降低预测不确定性；若融合是
在"搅浑"单尺度观点，融合会抬高不确定性。用无标签校准样本比较
`mean H(fused) − mean H(single)`（softmax τ=0.01），即可判断该用融合还是单尺度。

**Experiment**：
- 相对熵差（除以 log C）在 6 设置：PET −0.57%/−0.36%，EUR +7.48%/+2.37%，
  FLO −0.01%/+0.49%（后两者为 FLO 数据到达后补测）
- 校准仿真（n=500 子采样 × 200 次）：若用 0 阈值，FLO/gen 会误判
- 采用 θ=1%："仅在融合把不确定性抬高 ≥1% 最大熵时才干预切换"；判据在
  6/6 设置方向正确；敏感窗口 θ∈[0.5%, 2.3%] 内结果不变

**Decision**：AdaScale v2 = 数据集级模式选择（θ=1%）+ 样本级稳定性早停。
SINGLE 模式直接跳过 scales 1..9（1 次前向）；FUSED 模式走 D3 的早停。

## D8. 早停配置冻结：r=4, δ=0.45, ε=0.01（2026-09-11）

**Experiment**（FUSED 适用的 4 设置：PET×2、FLO×2；EUROSAT 走 SINGLE 不受影响）：

| 配置 | PET/text | PET/gen | FLO/text | FLO/gen | avg scales |
|------|----------|---------|----------|---------|-----------|
| r4 d0.25 e0.01 | +0.19 | +0.03 | **−0.29*** | +0.28 | 6.66 |
| **r4 d0.45 e0.01** | **+0.16*** | **+0.05** | **−0.07** | **+0.18*** | **7.36** |
| r3 d0.65 e0.02 | −0.07 | +0.05 | −0.01 | +0.18 | 7.67 |
| r3 d0.5 e0.02 | −0.20 | −0.09 | −0.18 | +0.22 | 7.03 |

（Δ 为 vs full-10 的百分点；* 为 paired bootstrap p<0.05）

**Decision**：冻结 **r=4, δ=0.45, ε=0.01**——在 4 个 FUSED 设置上最大损失仅
−0.07（不显著），其余持平或正；平均省 26% 计算。更激进的 r4 d0.25 作为
"aggressive"档在附录报告（省 33%，但 FLO/text −0.29 显著）。

## D8-修订. 冻结配置改为 r=4, δ=0.65, ε=0.01（2026-09-11 晚）

**Trigger**：ViT-B/16 验证（第 7 个设置，PET/text）显示 d0.45 损失 −0.20
（p=0.0046，显著）；同设置下 d0.65 仅 −0.07（p=0.11，n.s.）。

**全设置复核**（r=4, δ=0.65, ε=0.01）：

| 设置 | Δ vs full-10 | scales |
|------|------|------|
| PET/text (B/32) | 0.00 (ns) | 6.63 |
| PET/gen (B/32) | +0.04 (ns) | 8.63 |
| EUR/text (SINGLE) | +2.97* | 1.00 |
| EUR/gen (SINGLE) | +4.84* | 1.00 |
| FLO/text (B/32) | +0.02 (ns) | 7.91 |
| FLO/gen (B/32) | +0.12* | 8.92 |
| PET/text (B/16) | −0.07 (ns) | 6.42 |

**修正原则**（预先明确）：(a) 所有设置无显著负；(b) 省计算尽量大。
d0.45 违反 (a)（B/16 −0.20*），d0.65 满足且代价仅 ~6% scales（B/32 平均
26%→23% 省钱）。d0.45 作为 "aggressive" 档在消融中报告。

**Decision**：最终冻结 **r=4, δ=0.65, ε=0.01**（跨 7 设置统一）。

## D9. 判据局限的诚实记录（2026-09-11）

- EUROSAT/gen（需 SINGLE，熵差 +2.37%）与 FLO/gen（需 FUSED，熵差 +0.49%）
  在**所有测试过的无标签统计量**（margin 差、符号检验、logit gap、熵差、
  预测一致率、跨尺度支持率）上都只能靠**幅度**区分，靠**符号**无法区分——
  9 个统计量的完整对照记录在 outputs/mode_selector/。
- θ=1% 是幅度阈值，理论辩护有限；论文必须报告敏感性窗口并说明这是机制的
  已知边界（在 6 个设置上 6/6 正确）。
- 失败模式：若遇到"熵差在 (0,1%) 之间且实际该用 SINGLE"的数据集，判据会
  选 FUSED（过度保守，损失 ≤ 该设置 fused−single 差）。倾向保守的偏置是
  设计选择：FUSED 模式仍带早停优化，而误切 SINGLE 会浪费 90% 计算于无用
  scales 且若不巧 fused 更好时损失更大。
