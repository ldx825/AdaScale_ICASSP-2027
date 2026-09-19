# Related Work Matrix & Citation Verification

日期：2026-02（novelty-mining 阶段）
规则：每篇引用在写入 bib 前验证 title/authors/venue/year/DOI（来源：arXiv API、Semantic Scholar API、CVF Open Access）。

## 验证通过的 7 篇 prior work

| # | Work | Authors | Venue/Year | ID | 验证来源 | 相关性 |
|---|------|---------|-----------|-----|---------|--------|
| 1 | AdapTTA: Adaptive Test-Time Augmentation for Reliable Embedded ConvNets | L. Mocerino, R. G. Rizzo, V. Peluso, A. Calimera, E. Macii | VLSI-SoC 2021 | DOI 10.1109/VLSI-SoC53125.2021.9606980 | Semantic Scholar | 按输入复杂度控制 TTA 前向次数；confidence-based stopping |
| 2 | Better Aggregation in Test-Time Augmentation | D. Shanmugam, D. Blalock, G. Balakrishnan, J. Guttag | ICCV 2021, pp.1214-1223 | arXiv 2011.11156 | CVF OpenAccess | TTA 平均可次优；TTA 可翻转正确预测 |
| 3 | Efficient improvement of classification accuracy via selective test-time augmentation | J. Son, S. Kang | Information Sciences 642:119148, 2023 | DOI 10.1016/j.ins.2023.119148 | Semantic Scholar | 预测不确定性决定是否 TTA（per-sample） |
| 4 | Intelligent Multi-View Test Time Augmentation | E. Ozturk, M. Prabhushankar, G. AlRegib | ICIP 2024, pp.617-623 | arXiv 2406.08593 | arXiv API + 页面 | 两阶段：选增强 + 不确定性阈值决定何时 TTA |
| 5 | Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models (TPT) | M. Shu, W. Nie, D.-A. Huang, Z. Yu, T. Goldstein, A. Anandkumar, C. Xiao | NeurIPS 2022 | arXiv 2209.07511 | arXiv API | CLIP；多增强视图；entropy 最小化 + confidence selection |
| 6 | Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning (DiffTPT) | C.-M. Feng, K. Yu, Y. Liu, S. Khan, W. Zuo | ICCV 2023, pp.2704-2714 | arXiv 2308.06038 | arXiv API | RRC/扩散增强；余弦相似度过滤视图；CLIP TPT |
| 7 | Adaptive test-time augmentation via KL-regularized reinforcement learning for robust visual inference | T. Mittal, A. Dubey, D. Saini, S. Yadav, A. Jain, A. Panwar, H. Namazi | Scientific Reports, 2026 | DOI 10.1038/s41598-026-62059-4 | Semantic Scholar | 样本级自适应增强；label-free；策略为学习得到（RL） |

## 概念对比矩阵（用于论文 Table 备选）

| Method | Training-free | Per-sample adaptive | Dynamic #views | Target-domain fusion diagnosis | Synthetic-prototype / CLIP |
|---|---:|---:|---:|---:|---:|
| AdapTTA (2021) | ✓ | ✓ | ✓ | ✗ | ✗ |
| Selective-TTA (2023) | ✓ | ✓ | partially (on/off) | ✗ | ✗ |
| Intelligent Multi-View TTA (2024) | ✓ | ✓ | ✓ | ✗ | ✗ |
| TPT (2022) | ✗ (prompt update) | ✓ (view selection) | ✓ | ✗ | ✓ |
| DiffTPT (2023) | ✗ (prompt update) | ✓ (view filtering) | ✓ | ✗ | ✓ |
| RL-based TTA (2026) | ✓(test-time label-free; policy learned offline) | ✓ | ✓ | ✗ | ✗ |
| **AdaScale (ours)** | ✓ | ✓ | ✓ | **✓** | **✓** |

注：两列 ✓ 的验证：
- AdapTTA：论文明确"adaptive decision on whether to perform additional TTA iterations"（per-image）。
- Selective-TTA：明确 per-sample 二元门控（应用/不应用 TTA）。
- Multi-View TTA：先选增强（stage 1）再阈值决定（stage 2）。
- RL-TTA 2026："the policy itself is learned"——training-free 一列标注为例外情形（测试时无标签、但训练了策略网络）。

## 与 AdaScale 的精确区分（论文用语）

> Prior adaptive-TTA methods allocate augmentation compute per sample, and prior aggregation work shows TTA can degrade correct predictions. In contrast, we study *synthetic-prototype multi-scale feature fusion* and identify a **target-domain-level fusion failure**: on some target domains the fused predictor is *systematically* inferior to the largest-scale predictor. AdaScale is a training-free hierarchical controller that first decides **whether fusion should be enabled for the domain at all** (label-free entropy diagnosis), and only then, per sample, how many scale views suffice.

## BibTeX（已入 paper/adascale.bib）

见 `paper/adascale.bib` 中 key：`mocerino2021adaptta`, `shanmugam2021better`, `son2023selective`, `ozturk2024intelligent`, `shu2022tpt`, `feng2023difftpt`, `mittal2026adaptive`。
