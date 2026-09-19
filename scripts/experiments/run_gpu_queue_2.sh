#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_2.sh — GPU 任务串行队列（stage 2，使用修复后的批处理脚本）
#   [A] PET 生成图多尺度特征（修复版重跑）
#   [B] EUROSAT 生成（SD 2.1，resume）
#   [C] EUROSAT 官方 single-scale
#   [D] EUROSAT 多尺度（修复版 batch）
#   [E] FLO 官方 single-scale
#   [F] FLO 多尺度（修复版 batch）
#   [G] FLO 生成
#   [H] FLO 生成图多尺度特征（修复版 batch）
# 用法: nohup bash scripts/experiments/run_gpu_queue_2.sh > logs/gpu_queue_2.log 2>&1 &
# =============================================================================
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"
GEN_ROOT="$PROJECT_ROOT/third_party/LG-CLIP/dataset/SD_gen"

step() { echo "[gpu2 $(date +%H:%M:%S)] START $1"; }
done_() { echo "[gpu2 $(date +%H:%M:%S)] DONE  $1 (exit=$?)"; }

# ---- [A] PET 生成图多尺度特征（修复版） ----
step "A: PET gen multi-scale features (fixed batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 \
    --mode gen --batch 128 --gen_root_path "$GEN_ROOT" --image_root "$PROJECT_ROOT/data" \
    > logs/extract_gen_PET_v2.log 2>&1
done_ "A"

# ---- [B] EUROSAT 生成（resume） ----
step "B: EUROSAT generation (resume)"
"$PY" src/analysis/sd_gen_fast.py --dataset EUROSAT --ngen 10 \
    --gen_root_path "$GEN_ROOT" > logs/gen_EUROSAT.log 2>&1
done_ "B"

# ---- [C] EUROSAT 官方 single-scale ----
step "C: EUROSAT single-scale (official)"
bash scripts/baseline/run_baseline.sh single EUROSAT ViT-B/32 > logs/single_EUROSAT.log 2>&1
done_ "C"

# ---- [D] EUROSAT ms（修复版 batch） ----
step "D: EUROSAT multi-scale (fixed batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset EUROSAT --backbone ViT-B/32 \
    --mode real --batch 128 > logs/ms_EUROSAT_batch.log 2>&1
done_ "D"

# ---- [E] FLO 官方 single-scale ----
step "E: FLO single-scale (official)"
bash scripts/baseline/run_baseline.sh single FLO ViT-B/32 > logs/single_FLO.log 2>&1
done_ "E"

# ---- [F] FLO ms（修复版 batch） ----
step "F: FLO multi-scale (fixed batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset FLO --backbone ViT-B/32 \
    --mode real --batch 128 > logs/ms_FLO_batch.log 2>&1
done_ "F"

# ---- [G] FLO 生成 ----
step "G: FLO generation"
"$PY" src/analysis/sd_gen_fast.py --dataset FLO --ngen 10 \
    --gen_root_path "$GEN_ROOT" > logs/gen_FLO.log 2>&1
done_ "G"

# ---- [H] FLO 生成图多尺度特征 ----
step "H: FLO gen multi-scale features (fixed batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset FLO --backbone ViT-B/32 \
    --mode gen --batch 128 --gen_root_path "$GEN_ROOT" --image_root "$PROJECT_ROOT/data" \
    > logs/extract_gen_FLO.log 2>&1
done_ "H"

echo "[gpu2] ALL DONE"
