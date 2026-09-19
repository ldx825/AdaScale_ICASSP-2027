#!/usr/bin/env bash
# =============================================================================
# run_postqueue_analysis.sh — 等待 GPU 队列完成后自动运行全部分析（v2: 幂等）
#   [1] EUROSAT/FLO text-proto audit + 模拟 + 图表（已存在则跳过）
#   [2] EUROSAT/FLO gen-proto audit + 模拟 + 图表（已存在则跳过）
#   [3] paired bootstrap（PET/EUROSAT/FLO）
#   [4] Runtime benchmark（PET: full/fixed-5/fixed-7/adascale）
# 用法: nohup bash scripts/experiments/run_postqueue_analysis.sh > logs/postqueue.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"

echo "[postqueue $(date +%H:%M:%S)] waiting for GPU queue to finish..."
for i in $(seq 1 600); do
  if grep -q "ALL DONE" logs/gpu_queue_2.log 2>/dev/null; then
    echo "[postqueue $(date +%H:%M:%S)] GPU queue DONE"
    break
  fi
  sleep 60
done

BK="ViT-B/32"
BK_DIR="ViT-B32"

run_audit() {  # $1=ds $2=proto
  local ds=$1 proto=$2
  local out="outputs/scale_audit/$ds/$BK_DIR/summary_$proto.json"
  if [ -f "$out" ]; then echo "[skip] audit $ds/$proto (exists)"; return 0; fi
  "$PY" scripts/audit/run_scale_audit.py --dataset "$ds" --backbone "$BK" --proto "$proto" \
    && "$PY" src/analysis/plot_audit.py --dataset "$ds" --backbone "$BK" --proto "$proto" \
    && "$PY" src/analysis/predict_future_flip.py --dataset "$ds" --backbone "$BK" --proto "$proto"
}

run_sim() {  # $1=ds $2=proto
  local ds=$1 proto=$2
  local out="outputs/methods/$ds/$BK_DIR/sim_results_$proto.csv"
  if [ -f "$out" ]; then echo "[skip] sim $ds/$proto (exists)"; return 0; fi
  "$PY" scripts/experiments/exp_simulation.py --dataset "$ds" --backbone "$BK" --proto "$proto" \
    && "$PY" scripts/experiments/plot_pareto.py --dataset "$ds" --backbone "$BK" --proto "$proto"
}

for ds in EUROSAT FLO; do
  echo "[postqueue $(date +%H:%M:%S)] === $ds text protocol ==="
  run_audit "$ds" text
  run_sim "$ds" text
done

for ds in EUROSAT FLO; do
  echo "[postqueue $(date +%H:%M:%S)] === $ds gen protocol ==="
  if ls data/$ds/*feature_gen10_ms.hdf5 >/dev/null 2>&1; then
    run_audit "$ds" gen
    run_sim "$ds" gen
  else
    echo "[postqueue] gen features missing for $ds, skip"
  fi
done

echo "[postqueue $(date +%H:%M:%S)] === paired bootstrap ==="
for ds in PET EUROSAT FLO; do
  for proto in text gen; do
    if ls outputs/stats/${ds}_${BK_DIR}_${proto}_*_paired.csv >/dev/null 2>&1; then
      echo "[skip] bootstrap $ds/$proto"; continue
    fi
    "$PY" scripts/final/paired_bootstrap.py --dataset "$ds" --backbone "$BK" --proto "$proto" \
      || echo "[warn] bootstrap $ds/$proto failed"
  done
done

echo "[postqueue $(date +%H:%M:%S)] === runtime benchmark (PET) ==="
mkdir -p outputs/efficiency
if ls outputs/efficiency/PET_adascale.json >/dev/null 2>&1; then
  echo "[skip] runtime benchmark (exists)"
else
  "$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode full \
      --batch 32 --warmup 2 --repeats 3 --out outputs/efficiency/PET_full.json
  "$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode fixed --k 5 \
      --batch 32 --warmup 2 --repeats 3 --out outputs/efficiency/PET_fixed5.json
  "$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode fixed --k 7 \
      --batch 32 --warmup 2 --repeats 3 --out outputs/efficiency/PET_fixed7.json
  "$PY" src/adascale/runtime_bench.py --dataset PET --backbone ViT-B/32 --mode adascale \
      --batch 32 --warmup 2 --repeats 3 --out outputs/efficiency/PET_adascale.json
fi

echo "[postqueue $(date +%H:%M:%S)] ALL DONE"
