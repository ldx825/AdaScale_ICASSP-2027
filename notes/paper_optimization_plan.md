# AdaScale Paper 优化执行文档（Agent Runbook）

> 日期：2026-09-15 ｜ 适用目录：`/root/autodl-tmp/EviZO-VP/AdaScale-CLIP/paper/`
> 本文档为**可独立执行**的优化指令集：每项含现状证据、处方（文件/函数/参数）、验收标准。
> 执行前环境：`cd AdaScale-CLIP && export HOME=$PWD/.home HF_HOME=$PWD/.cache/huggingface TORCH_HOME=$PWD/.cache/torch`；LaTeX 用 `latexmk -pdf -interaction=nonstopmode main.tex`。

---

## 0. 验收基线（全项目通用）

| 指标 | 当前值 | 优化后要求 |
|---|---|---|
| main.pdf 页数 | 5（4 正文 + 1 refs） | 不变：4 页正文 + 第 5 页 refs（第 5 页参考 85–100% 填充） |
| 编译错误 | 0 | 0 |
| 图 PDF 字体 | **Type 3（位图化，发虚的根因）** | **Type 42/TrueType（真矢量）**；`pdffonts x.pdf` 验证 |
| Overfull \hbox | 3 处（全部在主表，最大 22.2pt） | 表格 0 处 |
| 被引文献 | 14 条 | 26–28 条（全部真实，验证记录见 §6） |

检查命令：
```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
grep -c Error main.log && grep "Overfull" main.log | head
pdffonts figures/...pdf | head -5        # 查看字体类型
pdfinfo main.pdf | grep Pages
```

---

## 1. Figure 1（架构图）— 重画为顶会风格（交给画图 AI）

**现状**：`figures/architecture/fig_arch.pdf`，matplotlib 手绘，信息正确但视觉朴素。用户要求用画图 AI 重画（内容参考本文，风格参考 CVPR 顶会）。

**画图提示词见本文档附录 A（回复中同步给出）**。

**重要提醒（写入给执行者）**：
1. 生成式画图工具（Midjourney/DALL·E/GPT-4o 等）对**图中的文字**通常不准确——建议将 AI 输出视为**风格稿**，再用 Figma/Illustrator/Keynote 按同样风格精确重绘；或只取其配色/构图灵感。
2. 最终交付要求：**矢量 PDF/SVG**；若仅有位图，需 ≥4× 分辨率（4000px 宽）且图内无小字（小字必糊）。
3. 落位：替换 `paper/figures/architecture/fig_arch.pdf` 后编译；保持 `width=0.9\linewidth`（若新图比例不同，等比调整至单栏宽约 3.3in，高度变化 ±0.15in 内不破坏 4 页）。
4. **备选方案（若无法获得好的矢量图）**：增强现有 matplotlib 版——字号 7.4→8.2、线宽 0.8→1.0、圆角半径增大、配色改为低饱和（`#E8F1FB`/`#E6F4EA`/`#FEF3E2`）、去除 `\u2003` 特殊空格、输出前设 `rcParams['pdf.fonttype']=42` 与 `ps.fonttype=42` 强制 TrueType。脚本：`scripts/paper/fig_architecture.py`。

---

## 2. Figure 2（两 regime 前缀曲线）— 真矢量 + 更清晰

**现状证据**：`paper/figures/figA_pet_text.pdf` 与 `figA_eurosat_text.pdf` 内嵌字体为 **Type 3**（`pdffonts` 已证）——这就是"不够清晰"的技术根因（Type 3 字形是迷你位图，打印/缩放下发虚）。

**处方**（写一个新脚本 `scripts/paper/fig_regime.py`）：
1. 数据来源（与论文一致）：
   - prefix 曲线：`outputs/scale_audit/PET/ViT-B32/per_sample_text.parquet` 与 `.../EUROSAT/ViT-B32/per_sample_text.parquet`（列含 per-scale 正确性；亦可从 `summary_text.json` 的 `prefix_acc_curve` 取 10 点）
   - single-scale 参考线：same json 或 per_sample 里 scale-10 单视图正确率
2. 绘图规范（关键）：
```python
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42      # ← 核心：TrueType，消灭 Type 3
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.linewidth"] = 0.8
rcParams["font.size"] = 8
# 线：prefix 实线 + 圆点，single 虚线 + 方点；线宽 1.4；图尺寸 (3.4, 2.0)，双图并排或两个独立文件
```
3. 输出 `figures/figA_pet_text.pdf`、`figures/figA_eurosat_text.pdf`（覆盖），并同步复制到 `paper/figures/`。
4. **验收**：`pdffonts` 显示 `TrueType`；100% 打印缩放肉眼清晰；曲线数字与 main.tex §4.2 描述一致（PET 中间峰、EuroSAT 融合劣于单尺度）。

---

## 3. Figure 3（attention 案例图）— 更大 + 更清晰

**现状**：`paper/figures/deep_diag/fig_attn_eurosat.pdf`，跨栏 `width=0.70\textwidth`，图内标题字号 6pt，偏小。

**处方**：
1. 重生成（脚本已参数化）：修改 `scripts/analysis/fig_attn_case.py`：
   - `figsize=(1.02*8, 1.55)` → `(1.15*8, 1.9)`（每格更大）
   - `fontsize=6` → `9`（两个 set_title 处）
   - 增加：`rcParams["pdf.fonttype"]=42`（若注释里定义）
   - 重跑两条命令（EUROSAT/PET 案例，索引 15057 / 6562，见 `notes/figure_plan.md`）
2. main.tex 中 `\includegraphics[width=0.70\textwidth]...fig_attn_eurosat.pdf` → **`0.90\textwidth`**
3. **页数补偿**：放大后预计多占 ~0.1 页——在 §4.3 main results 段落压缩 2–3 行（如将 "(paired bootstrap, no setting significantly negative, Table~\ref{tab:main})" 精简为 "(paired bootstrap)"，省 1 行；将 Fixed-5 句 "significantly beats fixed-budget truncation (Fixed-5: +1.2 to +18.5 points, all p<1e-3)" 缩为 "(Fixed-5: +1.2..+18.5, p<1e-3)"），或复用文末 `\enlargethispage{2.5\baselineskip}`（若 overflow ≤2 行可调至 3）。
4. **验收**：编译 0 error、仍 4 页正文 + refs 页；图中每格 100% 打印可读。

---

## 4. Table 2（主表，`paper/tables/tab_main.tex`）— 修复溢出

**现状证据**：`main.log` 中 3 处 `Overfull \hbox`（22.16pt / 21.29pt / 5.40pt），全部由 `tab_main.tex` 引发——**第 8 列（#S）挤出栏右边界，视觉上"挤到另一栏"**。

**根因**：8 列 × 内容宽（`Oxford-Pets`、`Flowers-102`、"Full-10/Single-10" 长列名）+ `\tabcolsep=3pt` 超出单栏（≈3.34in）。

**处方（改 `scripts/paper/build_tables.py` 的 `build_main()`，勿手改 tex）**，按序尝试：
1. `\setlength{\tabcolsep}{2pt}`（3pt→2pt，省 ~14pt）；
2. 列名缩短：`Oxford-Pets`→`Pets`、`Flowers-102`→`Flowers`、`CUB-200`→`CUB`（caption 注明 "Pets = Oxford-Pets, Flowers = Flowers-102"）、`Single-10`→`Single`、`\#S`→`$\bar{s}$`（avg scales，caption 说明）；
3. 如仍溢出：`\small`→`\footnotesize`（同时保留 2pt）；
4. 数值宽度一致性：`\textbf{82.63}` 保持，`$\Delta$` 列 `+0.18` 不加多余空格。
**验收**：`grep -c "Overfull" main.log`（针对该表）为 0；渲染后 #S 列完整；其余表格（tab_baselines 8 列同法自查：若其 also overfull，同样处理——当前 log 未见其为 overfull 源，但放大检查一遍）。
**注意**：改完重跑 `python scripts/paper/build_tables.py` 再编译。

---

## 5. Figure 4（机制双联图）— 真矢量 + 更清晰

**现状证据**：`paper/figures/deep_diag/figM_compact.pdf` 同样是 **Type 3** 字体（`pdffonts` 已证）。

**处方**（改 `scripts/analysis/plots_deep_diag.py` 的 `fig_mechanism_compact()`）：
1. 文件头部加：
```python
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
```
2. 字号放大：标题 7.5→9；刻度/图例 6.2→7.5；柱顶数字 6.2→7.5；`figsize=(7.0, 2.35)`→`(7.0, 2.7)`；
3. 重跑：`python scripts/analysis/plots_deep_diag.py` → 覆盖 `figures/deep_diag/figM_compact.pdf`，复制到 `paper/figures/deep_diag/`；
4. main.tex 中 `width=0.86\linewidth` 可微调（0.86–0.95），若增加高度需从 §4.4 baselines 段压缩 2 行（如 "confidently-wrong prefix passes any margin threshold, entrenching the domain-level failure" 缩短）。
**验收**：`pdffonts` 为 TrueType；方框中 (a)(b) 面板标注、数字（97/93/85/76/95/92）在 100% 打印下清晰可读；页数不变。

---

## 6. 参考文献扩充（补满第 5 页）——已验证引用池

**现状**：被引 14 条；第 5 页 refs 约占 50 行（~半页）。目标：**26–28 条**，填满第 5 页（双栏 ~90–100 行容量）。

### 6.1 已验证新增引用（12 条，全部经 arXiv 官方页面 / Semantic Scholar 核实）

> 验证时间：2026-09-15。**每条均核对了标题、作者、发表venue、arXiv ID 的一致性**。你只需把下方 BibTeX 追加到 `paper/adascale.bib`，并在正文相应位置 `\cite`。

```bibtex
@inproceedings{zhou2022cocoop,
  title     = {Conditional Prompt Learning for Vision-Language Models},
  author    = {Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@inproceedings{zhang2022tipadapter,
  title     = {Tip-Adapter: Training-Free Adaption of {CLIP} for Few-Shot Classification},
  author    = {Zhang, Renrui and Fang, Rongyao and Gao, Peng and Li, Kunchang and Dai, Jifeng and Qiao, Yu and Li, Hongsheng},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}

@article{gao2024clipadapter,
  title   = {{CLIP-Adapter}: Better Vision-Language Models with Feature Adapters},
  author  = {Gao, Peng and Geng, Shijie and Zhang, Renrui and Ma, Teli and Fang, Rongyao and Zhang, Yongfeng and Li, Hongsheng and Qiao, Yu},
  journal = {International Journal of Computer Vision (IJCV)},
  year    = {2024}
}

@inproceedings{sun2020ttt,
  title     = {Test-Time Training with Self-Supervision for Generalization under Distribution Shifts},
  author    = {Sun, Yu and Wang, Xiaolong and Liu, Zhuang and Miller, John and Efros, Alexei A. and Hardt, Moritz},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@inproceedings{wang2021tent,
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author    = {Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{zhang2022memo,
  title     = {{MEMO}: Test Time Robustness via Adaptation and Augmentation},
  author    = {Zhang, Marvin and Levine, Sergey and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{caron2020swav,
  title     = {Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author    = {Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}

@inproceedings{touvron2019fixres,
  title     = {Fixing the Train-Test Resolution Discrepancy},
  author    = {Touvron, Hugo and Vedaldi, Andrea and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}

@article{zhou2022coop,
  title   = {Learning to Prompt for Vision-Language Models},
  author  = {Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal = {International Journal of Computer Vision (IJCV)},
  year    = {2022}
}

@inproceedings{khattak2023maple,
  title     = {{MaPLe}: Multi-Modal Prompt Learning},
  author    = {Khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

@article{xian2019zsl,
  title   = {Zero-Shot Learning---A Comprehensive Evaluation of the Good, the Bad and the Ugly},
  author  = {Xian, Yongqin and Lampert, Christoph H. and Schiele, Bernt and Akata, Zeynep},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2019}
}
```

> 注：`zhou2022coop` 已存在于 adascale.bib（本清单中的版本为 IJCV 正式版信息，建议以本版覆盖）；若 `adascale.bib` 里已有其他重复 key，保留一个即可。

### 6.2 数据集引用（bib 里已存在、正文未引——直接加 `\cite` 即可；内容属领域常识级，但仍建议人工最终过目一遍原条目）

- `eurosat2019` → Helber et al., IEEE JSTARS 2019（EuroSAT）— 已核实 arXiv:1709.00029
- `pets2012` → Parkhi et al., BMVC 2012（Oxford-IIIT Pet）
- `flowers2008` → Nilsback & Zisserman, BMVC Workshop 2008（Flowers-102）
- `cub2011` → Wah et al., Caltech Tech Report 2011（CUB-200-2011）

### 6.3 引用落位（每处加 1–3 条，保持语义自然）

| 位置 | 建议 cite | 措辞示例 |
|---|---|---|
| Intro 段 2（自适应 TTA 背景）| `sun2020ttt`, `wang2021tent`, `zhang2022memo` | "test-time adaptation updates models on unlabeled test data [Refs], while our controller is training-free" |
| Related §2.3（CLIP 适配）| `zhou2022coop`, `zhou2022cocoop`, `gao2024clipadapter`, `zhang2022tipadapter`, `khattak2023maple` | "prompt- and adapter-based CLIP adaptations [Refs]" |
| Related §2.2（多尺度/视图）| `caron2020swav` | "multi-crop augmentation [ref] is standard in representation learning; we instead adapt how many crops are evaluated" |
| Related §2.3 或 §2.1（尺度不匹配）| `touvron2019fixres` | "train–test scale mismatch [ref] confirms scale matters for recognition" |
| Setup（数据集）| `eurosat2019`, `pets2012`, `flowers2008`, `cub2011` | "Datasets: Oxford-Pets [ref], EuroSAT [ref], Flowers-102 [ref] and CUB-200-2011 [ref]" |
| Related §2.1（零样本背景）| `xian2019zsl` | "zero-shot recognition [ref] …" |

**页数控制**：新增 12+4 条后如 refs 超出第 5 页 → 优先删 `xian2019zsl`、`khattak2023maple`（点缀性），其次合并引用；**严禁为了填满而引入任何未验证文献**。

### 6.4 防虚空引用 · 验证清单（执行者逐条核对）

- [ ] 新增 12 条 bibtex 的 arXiv ID 与标题对应（对照下表——本表为 2026-09-15 实际核验记录）；
- [ ] 全库检查：所有 `\cite{key}` 在 bib 中存在（编译后无 "Citation undefined" 警告）；
- [ ] bib 中存在的但正文不引用的条目（如 `imagenethard2023`）不影响编译，但**建议删除或引用**以保持整洁；
- [ ] 不新增任何未经本表记录的文献。

| key | arXiv ID | 标题（核验一致） | Venue |
|---|---|---|---|
| zhou2022cocoop | 2203.05557 | Conditional Prompt Learning for Vision-Language Models | CVPR 2022 |
| zhang2022tipadapter | 2207.09519 | Tip-Adapter | ECCV 2022 |
| gao2024clipadapter | 2110.04544 | CLIP-Adapter | IJCV |
| sun2020ttt | 1909.13231 | Test-Time Training with Self-Supervision | ICML 2020 |
| wang2021tent | 2006.10726 | Tent: Fully Test-Time Adaptation by Entropy Minimization | ICLR 2021 |
| zhang2022memo | 2110.09506 | MEMO: Test Time Robustness via Adaptation and Augmentation | NeurIPS 2022 |
| caron2020swav | 2006.09882 | SwAV (multi-crop) | NeurIPS 2020 |
| touvron2019fixres | 1906.06423 | Fixing the train-test resolution discrepancy | NeurIPS 2019 |
| zhou2022coop | 2109.01134 | Learning to Prompt for Vision-Language Models | IJCV 2022 |
| khattak2023maple | 2210.03117 | MaPLe: Multi-modal Prompt Learning | CVPR 2023 |
| xian2019zsl | 1707.00600 | ZSL — Comprehensive Evaluation | TPAMI 2019 |
| eurosat2019 | 1709.00029 | EuroSAT dataset | IEEE JSTARS 2019 |

（本轮验证中**拦截了两个错误 ID**——`2103.13845` 实为天体物理论文、`2404.02509` 实为量子物理论文——均**未**进入引用池。此流程须保持。）

---

## 7. （本文件）

---

## 8. 论文叙事强化（摘要 & 引言）

**目标叙事流**：*我们发现了一个被忽视的现象（域级融合失效）→ 诊断其成因（无标签熵诊断）→ 提出零成本修复（层级控制器 + 软融合）→ 成本大降且成功率更高。*

### 8.1 摘要修改草案（替换现摘要中对应句子）

- 第 1–2 句保持（现象引入很好）。
- 将结果句升级为"成本+精度"双强调（现文本）：
  > "...AdaScale matches or exceeds the 10-scale baseline on all six settings---improving it significantly by up to 4.8 points---while reducing scale evaluations by up to 90\%, and it dominates prior-inspired adaptive-TTA baselines on the accuracy--compute Pareto front."

  建议改为（同时修正 "six settings"→"all evaluated settings；**注意 CUB 加入后是 seven**）：
  > "Across four datasets and both text and synthetic prototypes, AdaScale **cuts wall-clock inference cost by up to 8.8× and scale evaluations by up to 90\% while improving accuracy by up to 4.8 points**, making it the only method on the accuracy--compute Pareto front in both fusion-beneficial and fusion-harmful regimes."

- 在 "We propose AdaScale" 句后追加"优势"强调句：
  > "Crucially, AdaScale is entirely training-free and label-free: it adds no parameters, requires no prompt tuning, and its overhead is one calibration pass over unlabeled samples."

### 8.2 引言修改要点（§1）

1. 第 3 段（audit 描述）开头加**现象强调句**：
   > "We find a failure mode that has been overlooked in this pipeline: on some target domains, *adding* scales not only fails to help but makes the fused predictor systematically **worse** than a single large-scale view."
2. 第 5 段贡献 (iii) 改为**收益前置**句式：
   > "(iii) experiments on four datasets showing that AdaScale matches or exceeds the 10-scale baseline on every setting **while cutting scale evaluations by 10–90\% and wall-clock cost by up to 8.8×** — accuracy improves up to +4.8 points exactly where prior adaptive-TTA baselines collapse (up to −8.6 points)."
3. 保持 claim 安全：不写 "first"，不写 "newly discovered harm"（沿用 `notes/claim_audit.md` 口径）。
4. 检查全文 "six settings"→"seven settings"、"three datasets"→"four datasets" 是否全部更新（CUB 加入后遗留；当前 main.tex 摘要里仍是 "all six settings"，**需改**）。

### 8.3 验收

- 摘要 ≤ 200 词（ICASSP 无硬限但保持精炼）；
- 现象→方法→收益三段逻辑在摘要中一眼可辨；
- 所有数字与主表一致（+4.84/8.8×/90%/10–90%）。

---

## 9. 模板确认（ICASSP 2027）

**结论**：
1. 当前 `paper/spconf.sty` 是 **IEEE Signal Processing Society 官方样式文件**（文件头自述 "Style file for Signal Processing Society Conferences (ICASSP, ICIP)"，功能：175mm×226mm 版芯、双栏、编号标题、`\name`/`\address` 等）。**ICASSP 历年 Paper Kit 均以此为核心**，因此当前投稿版**在用官方模板族**是正确的。
2. **仍需核验（camera-ready 前必做）**：从 ICASSP 2027 官网作者区（Authors → Paper Kit / Author's Kit，通常经 Papercept/CMT 或 IEEE SPS 官网分发）下载当年 kit，核对三件事：
   - `spconf.sty` 与 kit 内版本一致（若有微调，直接替换）；
   - 页数规则（近年为 **4 页正文 + 第 5 页仅参考文献**，与当前排版一致）；
   - camera-ready 才需要的版权块/页眉（`\toappear{}` 等，投稿版无此项）。
3. **执行者注**：本次环境无法访问 ICASSP 2027 官网（未上线/网络限制），上述核验需在有网环境完成；除此之外**无需改动当前模板**。

---

## 10. 执行顺序与工作量（给执行 agent）

| # | 任务 | 依赖 | 预估 |
|---|---|---|---|
| 1 | §6 bib 扩充 + 落位 cite + 编译查页数 | 无 | 30–40 min |
| 2 | §2 图 2 重生成（fonttype=42）| 无 | 20 min |
| 3 | §5 图 4 重生成（fonttype=42 + 字号）| 无 | 20 min |
| 4 | §3 图 3 放大 + 页数补偿 | 无 | 20 min |
| 5 | §4 表 2 修复（build_tables.py）| 无 | 20 min |
| 6 | §8 摘要/引言改写 + "six→seven" 清理 | 无 | 30 min |
| 7 | §1 Fig 1 新图落位（等画图 AI 产出；备选 matplotlib 增强）| AI 出图 | — |
| 8 | 全量验收（§0 表格逐项） | 1–7 | 15 min |
| 9 | 提交：`git add paper/ scripts/ && git commit -m "paper polish: ..."` | 全部 | 5 min |

**提交规范**：每个大项完成即 commit（用户习惯小步提交），commit message 前缀如 `paper-polish:`。

---

## 附录 A · Figure 1 画图提示词（复制给画图 AI）

```
A clean, modern academic "method overview / pipeline" figure for a computer vision
paper, in the flat, airy style of top-conference (CVPR/ICCV/NeurIPS) method figures.
Flat vector illustration; soft pastel palette (light blue #E8F1FB, soft green #E6F4EA,
warm sand #FEF3E2, neutral grey #F2F2F2); rounded rectangles with thin dark-grey
outlines (1px); thin elegant arrows; generous white space; uppercase sans-serif
micro-labels; NO shadows, NO gradients, NO 3D, white background.

Composition (horizontal flow with one branch), max 6 boxes, short labels only:
1) Far left: small square photo thumbnail labeled "test image x" → an arrow to
   three nested squares of different sizes labeled "scale crops s1…s10".
2) Upper right: rounded blue box titled "Level 1 · Unlabeled Fusion-Utility
   Diagnosis", subtitle "compare mean entropy of fused vs. single predictor on
   unlabeled samples", with the formula "ΔH = H̄(fused) − H̄(single)".
3) Below it, a small orange diamond decision node: "ΔH > θ ?".
4) YES branch (left, soft green box): "SINGLE — encode largest scale only
   (1 forward pass)".
5) NO branch (center, soft blue box): "FUSED — sequential scale loop t = 1..10;
   stop when prediction is stable (agreement + margin + drift)"; draw a subtle
   circular arrow to indicate iteration.
6) Bottom: both branches merge into one wide neutral box:
   "Output — probability-level soft fusion: Σ_s s · softmax(f_s·P / τ)".

Style references (same flat pastel look and layout density):
- DINOv2, arXiv:2304.07193, Figure 1 (method overview)
- Segment Anything (SAM), arXiv:2304.02643, Figure 1
- BLIP-2, arXiv:2301.12597, Figure 2 (architecture overview)
```

> 生成后处理提示：AI 产出的文字多半不可直接使用，请按该风格在矢量工具中重绘（或仅采用其配色/构图）；最终导出 SVG/PDF 后放入 `paper/figures/architecture/`。

---

## 执行记录（2026-09-15 · 已落地）

**验收结果（实测）**
- `main.pdf`：**5 页**（4 页正文 + 第 5 页参考文献排满至页底）；0 error
- Overfull \hbox：**0 处**（修复前 3 处：22.16 / 21.29 / 5.40pt）
- 图字体：5 张主图全部 `CID TrueType`（Type 42；修复前 Type 3）
- 被引文献：14 → **23 条**（[1]–[23]），无虚空引用
- `supplement.pdf`：4 页，0 overfull（复查通过）

**逐项落地**

| # | 项目 | 实际做法 | 结果 |
|---|------|----------|------|
| 1 | Fig 1 | 备选方案已执行：matplotlib 版升级 Type42 矢量；等画图 AI 产出后替换 | ✅（最终图待 AI 版） |
| 2 | Fig 2 | 新建 `scripts/paper/fig_regime.py`（按显示尺寸设计 + Type42）；PET 融合 82.45 vs 单尺度 80.09；EuroSAT 融合 42.38 vs 单尺度 45.34 | ✅ |
| 3 | Fig 3 | `fig_attn_case.py` 重跑（画布 6.4in、字号 5.4、Type42）；宽度 0.70→**0.90\textwidth** | ✅（更大更清晰） |
| 4 | Table 2/3 | `build_tables.py`：colsep 2/2.5pt、列名 Pets/Flowers/CUB + caption 注释；audit/baselines 同步修宽 | ✅ 3→0 overfull |
| 5 | Fig 4 | `plots_deep_diag.py` 重设计（按显示尺寸 + Type42 + 90° 标签） | ✅ |
| 6 | 参考文献 | 实测第 5 页容量 ≈21–22 条 → 最终 **23 条**。保留新增：`sun2020ttt`、`wang2021tent`、`zhou2022coop`、`gao2024clipadapter`、`zhang2022tipadapter` + 4 条数据集（`eurosat2019/pets2012/flowers2008/cub2011`）。计划中的 `maple/xian/memo/swav/fixres/cocoop` 因容量未引（bib 保留备用） | ✅ 第 5 页排满 |
| 7 | 本文件 | 已完成 | ✅ |
| 8 | 叙事 | 摘要：six→seven、补 “8.8× 成本 + 反超 8.6 分崩溃”；引言：新增 “two failure regimes” 与 “parameter-updating TTA 正交、编码器冻结” 对比句；全文压缩 ≈10 行以保 4 页 | ✅ |
| 9 | 模板 | 无需改动（spconf 官方模板族）；camera-ready 前对照当年 Paper Kit 复核 | ✅ |

**排版备注（后续修改须知）**
- 第 4 页使用了 `\enlargethispage{3.5\baselineskip}`（文字底边距 ≈0.37in）；若再增删内容，先跑 §0 验收命令核对页数与边距；
- 修改任何图表请继续遵守：`pdf.fonttype=42`、按最终显示尺寸设计字号；
- 图/表生成脚本：`scripts/paper/{fig_regime,build_tables,fig_architecture}.py`、`scripts/analysis/{plots_deep_diag,fig_attn_case}.py`（改动后同步复制到 `paper/figures/`）。

---

## Round 2（同日 13:35–14:05 · 用户新图 + 引用页规则 + Fig4 修复）

| 项目 | 做法 | 结果 |
|------|------|------|
| Fig 1 替换 | 用户 AI 生成横幅图 → `figures/architecture/fig_arch_v3.png`（2048×768，嵌入约 375 dpi），改为跨栏 `figure*`、宽 `0.78\textwidth`、落位第 3 页顶部 | ✅ |
| 引用页规则 | 按 ICASSP"第 5 页仅含引用"严格执行：`\clearpage` 强制参考文献从第 5 页开始；正文 4 页不含引用；引用 23→**21 条**（去掉 `sun2020ttt`、`gao2024clipadapter` 的引用，bib 保留备用） | ✅ [1]–[21] 全部在第 5 页 |
| Fig 4 v4 | 图例移出坐标区（单行置于 (a) 标题下方）、轴标签缩短（`rescue (%)`）、字号放大（5.0–6.2pt）、`bbox_inches="tight"`、400 dpi | ✅ 重叠检测 0 collision pairs |
| 版面平衡 | 为容纳新 Fig 1 精修正文 ≈8 行；`\enlargethispage` 保留（第 4 页文字底边距 ≈0.72in） | ✅ 5 页、0 overfull、正文止于第 4 页底 |

> 核查提示：`pdfimages -list main.pdf` 可确认 Fig1 嵌入图为 2048×768（与 `fig_arch_v3.png` 哈希一致）；`pdftotext -f 5 -l 5` 应只见 `[1]`–`[21]`。

