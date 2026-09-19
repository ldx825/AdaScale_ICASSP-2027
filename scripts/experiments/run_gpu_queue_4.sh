#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_4.sh — 干净重跑 AdaScale bench（queue3 完成后，CPU 空闲时）
# 用法: nohup bash scripts/experiments/run_gpu_queue_4.sh > logs/gpu_queue_4.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"

echo "[gpu4 $(date +%H:%M:%S)] waiting for queue3..."
for i in $(seq 1 240); do
  grep -q "\[gpu3\] ALL DONE" logs/gpu_queue_3.log 2>/dev/null && break
  sleep 30
done
echo "[gpu4 $(date +%H:%M:%S)] queue3 done; clean re-run of AdaScale bench"
"$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode adascale \
    --r 4 --delta 0.45 --eps 0.01 --batch 32 --warmup 1 --repeats 3 \
    --out outputs/efficiency/PET_adascale_v2b.json > logs/bench_v2b.log 2>&1
echo "[gpu4 $(date +%H:%M:%S)] bench done (exit=$?)"
echo "[gpu4] ALL DONE"
