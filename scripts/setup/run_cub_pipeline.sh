#!/bin/bash
# CUB 数据集全链分析（在 vanilla_clip_ms.py 特征提取完成后运行）
# 用法: bash scripts/setup/run_cub_pipeline.sh
set -e
cd "$(dirname "$0")/../.."
export HOME=$PWD/.home HF_HOME=$PWD/.cache/huggingface TORCH_HOME=$PWD/.cache/torch TMPDIR=$PWD/.tmp

echo "===== [1/5] scale audit ====="
.venv/bin/python scripts/audit/run_scale_audit.py --dataset CUB --backbone ViT-B/32 --proto text 2>&1 | tail -5

echo "===== [2/5] AdaScale v2 evaluation ====="
.venv/bin/python scripts/final/evaluate_adascale_final.py --dataset CUB --proto text 2>&1 | tail -8

echo "===== [3/5] AdaScale v3 (soft fusion) ====="
.venv/bin/python scripts/final/adascale_v3_sim.py --dataset CUB --proto text 2>&1 | tail -6

echo "===== [4/5] paired bootstrap ====="
.venv/bin/python scripts/final/bootstrap_v2.py --dataset CUB --proto text --stab 4,0.65,0.01 2>&1 | grep -E "AdaScale|acc=" | tail -8

echo "===== [5/5] deep diagnostics ====="
.venv/bin/python scripts/analysis/deep_diag.py --dataset CUB --proto text 2>&1 | grep -E "quadrants|QQ|QN|NQ|NN|drag|early="

echo "===== DONE ====="
