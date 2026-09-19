#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — 运行官方 LG-CLIP baseline 脚本（带项目内环境隔离与日志）
#
# 用法:
#   bash scripts/baseline/run_baseline.sh single  PET ViT-B/32   # vanilla_clip.py
#   bash scripts/baseline/run_baseline.sh ms      PET ViT-B/32   # vanilla_clip_ms.py
#   bash scripts/baseline/run_baseline.sh single  EUROSAT ViT-B/32
#   bash scripts/baseline/run_baseline.sh ms      EUROSAT ViT-B/32
# =============================================================================
set -euo pipefail

MODE="$1"        # single | ms
DATASET="$2"
BACKBONE="$3"

PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
LGCLIP_DIR="$PROJECT_ROOT/third_party/LG-CLIP"
IMAGE_ROOT="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/outputs/baseline/$DATASET"
mkdir -p "$LOG_DIR"

# ---- 环境隔离 ----
export HOME="$PROJECT_ROOT/.home"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
export PYTHONPATH="$LGCLIP_DIR"          # 确保 using repo-local clip pkg
PY="$PROJECT_ROOT/.venv/bin/python"

cd "$LGCLIP_DIR"

if [ "$MODE" = "single" ]; then
    LOG="$LOG_DIR/vanilla_clip_${BACKBONE//\//}.log"
    echo "[run] vanilla_clip.py dataset=$DATASET backbone=$BACKBONE -> $LOG"
    "$PY" vanilla_clip.py --dataset "$DATASET" --backbone "$BACKBONE" \
        --image_root "$IMAGE_ROOT" --device cuda:0 2>&1 | tee "$LOG"
elif [ "$MODE" = "ms" ]; then
    LOG="$LOG_DIR/vanilla_clip_ms_${BACKBONE//\//}.log"
    echo "[run] vanilla_clip_ms.py dataset=$DATASET backbone=$BACKBONE -> $LOG"
    "$PY" vanilla_clip_ms.py --dataset "$DATASET" --backbone "$BACKBONE" \
        --image_root "$IMAGE_ROOT" --device cuda:0 2>&1 | tee "$LOG"
else
    echo "Unknown mode: $MODE (use single|ms)"; exit 1
fi

# 记录 commit
echo "[commit] $(git -C "$LGCLIP_DIR" rev-parse HEAD)" | tee -a "$LOG"
echo "[done] $MODE $DATASET $BACKBONE"
