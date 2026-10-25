# AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition

Code release for the paper **"AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition"** (manuscript under review).

- Paper PDF: [`paper/main.pdf`](paper/main.pdf) · Supplementary: [`paper/supplement.pdf`](paper/supplement.pdf)
- Authors: Dongxu Liu, Keming Fan, Xiang Luo, Chenrui Zhu, Dongyang Liu

This repository contains the code that implements the method and produces every number, table and figure in the paper. It is a research release: datasets, the LG-CLIP baseline checkout and the SD-generated prototype images are **not** distributed — see [Data preparation](#data-preparation).

---

## 1. Code overview

### 1.1 What the method does

AdaScale is a two-level, training-free and label-free controller on top of a frozen CLIP encoder that performs multi-scale test-time augmentation:

* **Level 1 — domain-level fusion diagnosis** (Sec. 3.3). On `n` unlabeled target samples it compares the mean normalized entropy of the fused 10-scale predictor with that of the largest-scale predictor, `ΔH = mean_i [H(p_fused) − H(p_single)] / log C`. If `ΔH > θ` the target domain is run in **Single** mode (one forward pass per image), otherwise in **Fused** mode.
* **Level 2 — sample-level reliability stopping** (Sec. 3.4). Inside Fused mode a sample exits the scale loop after scale `t ≥ r` when three conditions hold: top-1 agreement over the last `r` scales, softmax margin `m_t ≥ δ`, and JS drift `D_JS(p_t ‖ p_{t−1}) ≤ ε`.
* **Output rule** (Sec. 3.5). Evaluated views are fused in **probability space**, `p_out ∝ Σ_{s∈V} w_s · softmax(f_sᵀP / τ_s)` with `w_s = s` and `τ_s = 0.02`, instead of averaging features.

Everything is training-free: no parameter is fitted, no label is used, and no per-dataset tuning is performed. The frozen configuration used for all reported results is `r = 4`, `δ = 0.65`, `ε = 0.01`, `θ = 0.01`, `τ_s = 0.02`, `n = 500`.

### 1.2 Where the method lives

| Module | Role |
|---|---|
| [`src/adascale/adascale.py`](src/adascale/adascale.py) | The policy itself, exposed as a stateless `AdaScalePolicy.decide(t, idx, feat, logits, prob, js_prev, top1_hist)` interface (agreement + margin + JS-drift exit rule). |
| [`src/adascale/simulate.py`](src/adascale/simulate.py) | Mechanism-faithful simulator of dynamic multi-scale execution: samples accumulate scales in order, exited samples stop consuming scales and keep the prediction of their exit point. Also hosts the `FixedKPolicy` / `MarginOnlyPolicy` ablations and the prefix fusion (`w_s = s`, L2-normalised). This is the code path behind all reported accuracy / average-scale numbers. |
| [`src/adascale/runtime_bench.py`](src/adascale/runtime_bench.py) | Wall-clock benchmark of the real dynamic execution with active-set batching (batch 32, 2 warm-up iterations, 3 timed repetitions, `torch.cuda.synchronize()` around the timed region). |
| [`src/utils/load_features.py`](src/utils/load_features.py) | HDF5 IO for the per-scale feature cache and the aggregation primitives: `SCALE_WEIGHTS` (`w_s = s`), `prefix_aggregate()` (`Σ_{s≤t} w_s f_s`, L2-normalised) and `full_scale_aggregate()`. |
| [`src/analysis/prefix_analysis.py`](src/analysis/prefix_analysis.py) | Core audit statistics: prefix-accuracy curve, fraction of samples harmed by late scales, minimum equivalent scale `t_eq`, running-margin ROC-AUC. |
| [`src/analysis/extract_ms_batched.py`](src/analysis/extract_ms_batched.py) | Batched re-implementation of the reference multi-scale extraction that **preserves the reference RNG consumption order** (no `DataLoader`, manual per-sample crops, chunked forward passes) — needed for bit-comparable features. |
| [`src/analysis/plot_audit.py`](src/analysis/plot_audit.py), [`case_studies.py`](src/analysis/case_studies.py), [`predict_future_flip.py`](src/analysis/predict_future_flip.py), [`sd_gen_fast.py`](src/analysis/sd_gen_fast.py) | Audit figures, qualitative case studies, per-scale flip predictability analysis, and SD prototype generation with the reference parameters. |

### 1.3 Paper ↔ code correspondence

| Paper | Code |
|---|---|
| Sec. 3.1, Eq. (1) — multi-scale fusion with triangular weights `w_s = s/Σt`, `τ⁻¹ = 100` | `src/utils/load_features.py` (`SCALE_WEIGHTS`, `prefix_aggregate`, `full_scale_aggregate`); `LOGIT_TAU` in `src/adascale/*.py` |
| Sec. 3.2 / Table 1 — per-scale utility audit (prefix accuracy, harmed fraction, `t_eq`, margin AUC) | `scripts/audit/run_scale_audit.py` → `src/analysis/prefix_analysis.py` → `src/analysis/plot_audit.py` |
| Sec. 3.3, Eq. (2) — unlabeled fusion-utility diagnosis (`ΔH` vs `θ`) | `scripts/final/evaluate_adascale_final.py` (calibration + mode decision); probes and sensitivity: `scripts/final/mode_selector_probe.py`, `mode_calibration_sim.py`, `arbitration_probe.py` |
| Sec. 3.4 — scale-wise reliability stopping (agreement / margin / drift) | `src/adascale/simulate.py::StabilityPolicy`, `src/adascale/adascale.py::AdaScalePolicy` |
| Sec. 3.5, Eq. (3) — probability-level soft fusion (`w_s = s`, `τ_s = 0.02`) | `scripts/final/adascale_v3_sim.py` (candidate `F7`: per-view `softmax(·/0.02)`, triangular weights, applied on the frozen exit trajectory) |
| Sec. 4.2 — scale utility results, harmful late scales, EuroSAT fusion-is-worse regime | `scripts/audit/run_scale_audit.py`, `scripts/final/fusion_vs_single.py`, `scripts/final/single_scale_baselines.py` |
| Sec. 4.3 / Table 2 — main results (4 datasets × 2 prototype types) | `scripts/final/evaluate_adascale_final.py` + `adascale_v3_sim.py`, aggregated by `scripts/final/collect_results.py` |
| Sec. 4.4 — comparison with prior-inspired adaptive baselines | `scripts/final/prior_baselines.py` |
| Sec. 4.5 — ablations (Fixed-K vs AdaScale, `r/δ/ε` sensitivity) | `scripts/experiments/exp_simulation.py`, `fine_grid.py`, `plot_sensitivity.py`, `plot_pareto.py` |
| Sec. 4.6 — runtime / scale-evaluation savings | `src/adascale/runtime_bench.py` |
| Significance of every reported gap (paired bootstrap) | `scripts/final/bootstrap_v2.py`, `scripts/final/paired_bootstrap.py` |
| Suppl. — selector sensitivity (`n`, `θ` robustness), seed replication, deep diagnostics | `scripts/final/mode_calibration_sim.py`, `scripts/final/seed_replication.py`, `scripts/analysis/deep_diag.py`, `scripts/analysis/attention_maps.py`, `scripts/analysis/attn_stats.py` |

### 1.4 Repository layout

```
src/                 method (adascale/), audit statistics and feature IO (analysis/, utils/)
scripts/
  baseline/          wrappers around the reference LG-CLIP baseline scripts
  audit/             scale-utility audit + RNG-equivalence diagnostics
  experiments/       Fixed-K vs AdaScale simulation, Pareto / sensitivity plots, GPU queues
  final/             final pipeline: mode selector, early exit, soft fusion, baselines, bootstrap
  setup/             dataset / split builders (PET, EuroSAT, FLO, CUB)
environment/         frozen pip environment used for the reported runs
paper/               compiled paper PDF and supplementary PDF
outputs/, notes/     (not tracked) local experiment artefacts and research notes
```

---

## 2. Environment

Reported runs used Python 3.12, PyTorch 2.8.0+cu128, `torchvision`, `h5py`, `numpy`, `scipy` and `matplotlib` on a single RTX 4080 SUPER (32 GB). The full frozen environment is in [`environment/pip_freeze.txt`](environment/pip_freeze.txt).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/pip_freeze.txt   # or install torch/torchvision/h5py/numpy/scipy/matplotlib
```

The scripts resolve the project root from `Path(__file__).resolve().parents[2]`, so they can be run from any working directory inside a checkout.

## 3. Data preparation

Datasets and the CLIP checkpoint are not distributed.

| Dataset | Source | Expected location | Test images |
|---|---|---|---|
| Oxford-Pets (PET) | official Oxford-IIIT Pets archive | `data/PET/OxfordPets/{images,annotations}` | 7390 (all) |
| EuroSAT | HuggingFace `blanchon/EuroSAT_RGB` (parquet) | `data/EUROSAT/2750/<Class>/` | 27000 (all) |
| Flowers-102 (FLO) | official Oxford 102 Flowers archive + `labels.mat` | `data/FLO/Flowers102/{jpg,cat_to_name.json}` | 8189 (all) |
| CUB-200-2011 | official archive (text prototypes only) | `data/CUB/...` | 5794 |

Splits follow the reference protocol: **test = all original images**; the `train` split (5 images per class) only exists to satisfy the reference loader and never enters any evaluation. The reference repository ships no split files, so they are generated here:

```bash
python scripts/setup/build_pet_split.py
python scripts/setup/build_eurosat.py        # rebuild EuroSAT from the parquet files
python scripts/setup/build_flo_split.py
bash   scripts/setup/run_cub_pipeline.sh
```

Per-scale features are cached as HDF5 following the reference layout, and every downstream step reads only that cache:

```
data/<dataset>/multi_scale/CLIP_<backbone>_feature_scale{1..10}.hdf5   # test_f (N,D), test_l, all_embeddings
data/<dataset>/CLIP_<backbone>_feature_ms.hdf5                        # reference 10-scale fusion
data/<dataset>/multi_scale_gen/..._feature_gen10_scale{1..10}.hdf5    # synthetic prototypes
```

## 4. Reproducing the paper

`scripts/baseline/run_baseline.sh` wraps the reference LG-CLIP scripts. It contains the absolute project root of the machine used for the experiments (`PROJECT_ROOT=...`, `LGCLIP_DIR=...`); set both to your checkout before running.

```bash
# 0) reference baseline: single scale (text prototypes) and the official 10-scale fusion
bash scripts/baseline/run_baseline.sh single PET ViT-B/32
bash scripts/baseline/run_baseline.sh ms     PET ViT-B/32

# 1) scale-utility audit (text / synthetic prototypes)
#    first re-verifies that prefix-10 fusion matches the reference *_feature_ms.hdf5
python scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/32 --proto text
python src/analysis/plot_audit.py       --dataset PET --backbone ViT-B/32 --proto text

# 2) Fixed-K vs AdaScale simulation + accuracy-compute Pareto plot
python scripts/experiments/exp_simulation.py --dataset PET --backbone ViT-B/32 --proto text
python scripts/experiments/plot_pareto.py    --dataset PET --backbone ViT-B/32 --proto text

# 3) final method: Level-1 mode selection + Level-2 early exit
python scripts/final/evaluate_adascale_final.py --dataset PET --proto text --stab 4,0.65,0.01 --theta 0.01

# 4) probability-level soft fusion on the frozen exit trajectory (paper's output rule)
python scripts/final/adascale_v3_sim.py --dataset PET --proto text --stab 4,0.65,0.01

# 5) paired bootstrap significance + aggregation over all settings
python scripts/final/bootstrap_v2.py  --dataset PET --proto text --stab 4,0.65,0.01
python scripts/final/collect_results.py
```

`--stab r,delta,eps` selects the early-exit configuration; the paper's frozen configuration is `4,0.65,0.01` and is passed explicitly here, because the argparse defaults of a few analysis scripts still carry development values. Use `--proto gen` for the synthetic-prototype protocol.

Optional analyses: `scripts/final/prior_baselines.py` (adaptive-TTA baselines), `scripts/final/single_scale_baselines.py` and `fusion_vs_single.py` (fusion-vs-single regime), `scripts/final/mode_calibration_sim.py` (selector sensitivity), `scripts/final/seed_replication.py` (crop-seed robustness), `src/adascale/runtime_bench.py` (wall-clock), `scripts/audit/verify_batch_equiv.py` (feature-extraction equivalence check).

All intermediate results are written under `outputs/` (`outputs/scale_audit/`, `outputs/methods/`, `outputs/final/`, `outputs/efficiency/`, ...), which is not tracked in git.

## 5. Notes and limitations

* The reference LG-CLIP checkout (`third_party/LG-CLIP`, commit `e0a3c6f`) is not included; see `scripts/baseline/run_baseline.sh` for the expected path. Its `clip/bpe_simple_vocab_16e6.txt.gz` is missing upstream and must be added.
* Synthetic prototypes were generated with the plain class-label path of the reference SD pipeline (fixed seed 2024); the LLM-prompt path needs an external API key and was not used. Pixel-identical generation is therefore not guaranteed, which can shift the synthetic-prototype numbers slightly.
* Any `DataLoader` wrapper changes the RNG stream and therefore the random crops; use `src/analysis/extract_ms_batched.py` (no `DataLoader`) when re-extracting features.
* Exit behaviour is evaluated by simulating execution on the cached per-scale features (same crop pool, paired comparison); `src/adascale/runtime_bench.py` additionally measures the real dynamic execution.

## 6. Citation

```bibtex
@misc{liu2026adascale,
  title  = {AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition},
  author = {Liu, Dongxu and Fan, Keming and Luo, Xiang and Zhu, Chenrui and Liu, Dongyang},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 7. Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
