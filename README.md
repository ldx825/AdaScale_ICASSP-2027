# AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition

Official code and paper materials for the **ICASSP 2027** submission.

> **Code:** https://github.com/ldx825/AdaScale_ICASSP-2027 · **Paper PDF:** [`paper/main.pdf`](paper/main.pdf) · **Supplement:** [`paper/supplement.pdf`](paper/supplement.pdf)

**Authors:** Dongxu Liu, Keming Fan, Xiang Luo, Chenrui Zhu, Dongyang Liu (Jilin University / UESTC)

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | ICASSP 2027 LaTeX source, figures, compiled `main.pdf` and `supplement.pdf` |
| `scripts/` | Full reproduction pipeline: LG-CLIP baseline wrappers, scale-utility audit, Fixed-K vs AdaScale simulation/evaluation, paired bootstrap significance, paper table generation |
| `src/` | Method implementation (`adascale/`), analysis utilities (`analysis/`), feature-cache IO |
| `environment/` | Environment, dataset construction and reproducibility notes (`reproducibility.md`) |
| `outputs/` | Small result files (JSON/CSV summaries) and figures backing the paper tables |
| `notes/` | Research notes: method decisions, scale-utility findings, failed ideas |

Datasets (Oxford-Pets / EuroSAT / Flowers-102 / CUB-200), the LG-CLIP baseline checkout (`third_party/LG-CLIP`)
and SD-generated images are **not included**; see `environment/reproducibility.md` for setup details.

## Quick start

```bash
# Python environment: PyTorch 2.8.0+cu128 (details in environment/reproducibility.md)
export HOME=$PWD/.home HF_HOME=$PWD/.cache/huggingface TMPDIR=$PWD/.tmp

# 1) Reproduce the LG-CLIP baseline (example: PET, ViT-B/32)
bash scripts/baseline/run_baseline.sh single PET ViT-B/32   # single-scale, text prototypes
bash scripts/baseline/run_baseline.sh ms    PET ViT-B/32    # official 10-scale cache

# 2) Scale Utility Audit (text / gen protocols)
python scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/32 --proto text
python scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/32 --proto gen

# 3) Method evaluation (Fixed-K vs AdaScale simulation + Pareto plot)
python scripts/experiments/exp_simulation.py --dataset PET --backbone ViT-B/32 --proto text
python scripts/experiments/plot_pareto.py    --dataset PET --backbone ViT-B/32 --proto text

# 4) Final method (v2) + paired bootstrap + aggregation
python scripts/final/evaluate_adascale_final.py --dataset PET --proto text --stab 4,0.45,0.01
python scripts/final/bootstrap_v2.py --dataset PET --proto text
python scripts/final/collect_results.py     # -> outputs/final/all_results_v2.csv
```

## Compiling the paper

```bash
python scripts/paper/build_tables.py   # regenerate tables/*.tex
cd paper
latexmk -pdf main.tex                  # or: pdflatex + bibtex + pdflatex x2
latexmk -pdf supplement.tex
```

ICASSP template files are included (`spconf.sty`, `IEEEbib.bst`).

## Citation

```bibtex
@inproceedings{liu2027adascale,
  title     = {AdaScale: Diagnosing and Adapting Multi-Scale Fusion for Training-Free Zero-Shot Recognition},
  author    = {Liu, Dongxu and Fan, Keming and Luo, Xiang and Zhu, Chenrui and Liu, Dongyang},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
