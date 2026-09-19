#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_8.sh — 最终配置（r4, d0.65）的干净运行时基准
# 排在 queue7 之后（避免 GPU/CPU 竞争影响计时）
# 用法: nohup bash scripts/experiments/run_gpu_queue_8.sh > logs/gpu_queue_8.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"

echo "[gpu8 $(date +%H:%M:%S)] waiting for queue7..."
for i in $(seq 1 300); do
  grep -q "\[gpu7\] ALL DONE" logs/gpu_queue_7.log 2>/dev/null && break
  sleep 60
done
echo "[gpu8 $(date +%H:%M:%S)] queue7 done; running final-config bench (r4,d0.65)"
"$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode adascale \
    --r 4 --delta 0.65 --eps 0.01 --batch 32 --warmup 1 --repeats 3 \
    --out outputs/efficiency/PET_adascale_v2c.json > logs/bench_v2c.log 2>&1
echo "[gpu8 $(date +%H:%M:%S)] bench done (exit=$?)"
echo "[gpu8] ALL DONE"
