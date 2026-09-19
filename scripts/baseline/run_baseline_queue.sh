#!/usr/bin/env bash
# =============================================================================
# run_baseline_queue.sh — 串行运行剩余数据集的官方 baseline（single + ms）
# 输出写日志文件（不刷屏）。用法: bash scripts/baseline/run_baseline_queue.sh [datasets...]
# =============================================================================
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(EUROSAT FLO)
fi

for ds in "${DATASETS[@]}"; do
    for mode in single ms; do
        echo "[queue] $(date +%H:%M:%S) START $mode $ds ViT-B/32"
        bash scripts/baseline/run_baseline.sh "$mode" "$ds" ViT-B/32 \
            > "logs/queue_${mode}_${ds}.log" 2>&1
        rc=$?
        echo "[queue] $(date +%H:%M:%S) DONE  $mode $ds (exit=$rc)"
    done
done
echo "[queue] ALL DONE"
