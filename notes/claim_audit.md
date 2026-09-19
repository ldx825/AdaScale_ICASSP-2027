# Claim Audit（claim_audit.md）

日期：2026-02。规则：对论文（旧版 + 新版草稿）中每条 novelty/性能主张分类：
`SAFE` / `NEEDS-CITATION` / `TOO-STRONG` / `DELETE`。

## 逐条审计

| # | 旧版/潜在草稿句子 | 分类 | 处置 |
|---|---|---|---|
| 1 | "how many test-time scales an image actually needs has not been studied" | **DELETE** | AdapTTA 2021 / Selective-TTA 2023 已研究样本级视图分配。改为："Prior adaptive-TTA methods dynamically control augmentation usage or repeated inference, but do not address the target-domain-level fusion failure that arises in our synthetic-prototype multi-scale setting." |
| 2 | "We are the first to show that TTA can hurt predictions" | **DELETE** | ICCV 2021 (Better Aggregation) 已展示。改为："Consistent with prior observations that TTA aggregation can alter correct predictions, we find a distinct failure in synthetic-prototype scale fusion: on some target domains the fused predictor is systematically inferior to the largest-scale predictor." |
| 3 | "AdaScale is the first adaptive TTA framework" | **DELETE** | 不正确。 |
| 4 | "AdaScale: the first method to adaptively decide how many test-time scales an image needs" | **DELETE** | 同上。 |
| 5 | "We identify and quantify a target-domain-level fusion failure in synthetic-prototype multi-scale zero-shot recognition" | **SAFE** | 保留（"identify in our studied setting"表述，不写 first）。 |
| 6 | "We propose a label-free entropy diagnosis that decides whether multi-scale fusion should be enabled for a target domain" | **NEEDS-CITATION** | 保留 + cite Selective-TTA（样本级对照）、Multi-View TTA（两阶段对照）。 |
| 7 | "entropy gap *proves* fusion is harmful" | **TOO-STRONG** | 改为 "we use normalized entropy change as an empirical label-free proxy for fusion utility"。 |
| 8 | "Motivated by prior adaptive-TTA work, we instantiate a scale-wise reliability rule tailored to prefix multi-scale fusion" | **SAFE** | L2 的定位句（支持性机制，非新范式）。 |
| 9 | "AdaScale outperforms prior adaptive-TTA baselines at comparable compute" | **SAFE（有实验支持）** | B1/B2/B3/WGT 已跑；Pareto 分析见 baseline_comparison_plan.md。限定为 "at comparable or lower average scale counts"。 |
| 10 | "state-of-the-art zero-shot recognition" | **DELETE** | 无防守性比较。 |
| 11 | "significantly faster"（仅凭 scale count） | **TOO-STRONG** | 改用真实 wall-clock（已有 PET 202s vs 290s 数据）+ "fewer encoder calls"。 |
| 12 | "early exit is a novel mechanism" | **DELETE** | 降级为支持机制；cite adaptive-TTA。 |
| 13 | "AdaScale matches or improves Full-10 accuracy while reducing scale evaluations" | **SAFE** | 与主表数据一致（6 settings：3 显著正、3 持平/微正）。 |
| 14 | "fusion utility is a target-domain property, whereas scale sufficiency is sample-dependent" | **SAFE** | 核心 framing 句。 |
| 15 | "B2 (Selective-TTA-style) fails on EuroSAT while AdaScale succeeds" | **SAFE（有实验支持）** | 44.93/39.46 < 45.34/41.17；表述为 "cannot recover the single-scale advantage"。 |

## v3（soft fusion）主张

| # | 句子 | 分类 | 处置 |
|---|---|---|---|
| 16 | "probability-level triangular soft fusion (τ=0.02) improves FUSED accuracy by ~0.2pt on average, training-free" | **SAFE** | 6 设置 +0.03~0.34；注明 FLO/text 为持平。 |
| 17 | "τ=0.02 is chosen a priori (softening single-view overconfident distributions)" | **SAFE** | 先验设定声明，防止 test-tuning 指控；不得称其为"最优温度搜索"。 |

## 禁止事项清单（spec §26）核对

- [x] 无编造数字——所有数字可追溯 outputs/
- [x] 未运行/未赢的 baseline 不写入
- [x] 无未经验证的 "first"
- [x] 模式选择失败的设置如实报告（无失败——6/6 正确；若出现必须报告）
- [x] 阈值未在测试标签上调优（θ=0.01 / r4,d0.65,e0.01 冻结自校准集；τ=0.02 先验）
- [x] 熵仅称为 proxy，未称为证明
- [x] 早退定位为支持机制，非新范式
