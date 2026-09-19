#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_7.sh — EUROSAT × ViT-B/16 跨骨干验证（等 queue5 完成后）
#   [1] EUROSAT ViT-B/16 10-scale 提取
#   [2] audit + 模拟 + AdaScale v2 评估 + bootstrap（text 协议）
# 用法: nohup bash scripts/experiments/run_gpu_queue_7.sh > logs/gpu_queue_7.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"

step() { echo "[gpu7 $(date +%H:%M:%S)] START $1"; }
done_() { echo "[gpu7 $(date +%H:%M:%S)] DONE  $1 (exit=$?)"; }

echo "[gpu7 $(date +%H:%M:%S)] waiting for queue5..."
for i in $(seq 1 240); do
  grep -q "DONE" logs/queue5.log 2>/dev/null && break
  sleep 60
done

step "1: EUROSAT ViT-B/16 multi-scale extraction"
"$PY" src/analysis/extract_ms_batched.py --dataset EUROSAT --backbone ViT-B/16 \
    --mode real > logs/ms_EUROSAT_vitb16.log 2>&1
done_ "1"

step "2: EUROSAT ViT-B/16 audit + eval (text)"
"$PY" scripts/audit/run_scale_audit.py --dataset EUROSAT --backbone ViT-B/16 --proto text
"$PY" scripts/final/evaluate_adascale_final.py --dataset EUROSAT --backbone ViT-B/16 \
    --proto text --stab 4,0.45,0.01
"$PY" scripts/final/bootstrap_v2.py --dataset EUROSAT --backbone ViT-B/16 --proto text
"$PY" scripts/experiments/exp_simulation.py --dataset EUROSAT --backbone ViT-B/16 --proto text
done_ "2"

echo "[gpu7] ALL DONE"
