#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_1.sh — GPU 任务串行队列（stage 1）
#   [0] 验证 batch 版与官方 scale-1 特征一致性（PET）
#   [1] PET 生成图多尺度特征提取（batch）
#   [2] EUROSAT 生成（SD 2.1, 100 张）
#   [3] EUROSAT 官方 single-scale
#   [4] EUROSAT 多尺度（batch 版）
#   [5] FLO 官方 single-scale
#   [6] FLO 多尺度（batch 版）
#   [7] FLO 生成（SD 2.1, 1020 张）
#   [8] FLO 生成图多尺度特征提取（batch）
# 用法: nohup bash scripts/experiments/run_gpu_queue_1.sh > logs/gpu_queue_1.log 2>&1 &
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

step() { echo "[gpu-queue $(date +%H:%M:%S)] START $1"; }
done_() { echo "[gpu-queue $(date +%H:%M:%S)] DONE  $1 (exit=$?)"; }

# ---- [0] batch 一致性验证（PET scale 1） ----
step "0: batch-vs-official verification (PET scale1)"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 \
    --mode real --scales 1 --batch 128 --out_tag _batchtest > logs/verify_batch_PET.log 2>&1
"$PY" scripts/audit/verify_batch_equiv.py --dataset PET --backbone ViT-B/32 --scale 1 \
    >> logs/verify_batch_PET.log 2>&1
done_ "0"

# ---- [1] PET 生成图多尺度特征 ----
step "1: PET gen multi-scale features (batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 \
    --mode gen --batch 128 --gen_root_path "$GEN_ROOT" --image_root "$PROJECT_ROOT/data" \
    > logs/extract_gen_PET.log 2>&1
done_ "1"

# ---- [2] EUROSAT 生成 ----
step "2: EUROSAT generation"
"$PY" src/analysis/sd_gen_fast.py --dataset EUROSAT --ngen 10 \
    --gen_root_path "$GEN_ROOT" > logs/gen_EUROSAT.log 2>&1
done_ "2"

# ---- [3] EUROSAT 官方 single-scale ----
step "3: EUROSAT single-scale (official)"
bash scripts/baseline/run_baseline.sh single EUROSAT ViT-B/32 > logs/single_EUROSAT.log 2>&1
done_ "3"

# ---- [4] EUROSAT ms（batch 版） ----
step "4: EUROSAT multi-scale (batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset EUROSAT --backbone ViT-B/32 \
    --mode real --batch 128 --num_workers 0 > logs/ms_EUROSAT_batch.log 2>&1
done_ "4"

# ---- [5] FLO 官方 single-scale ----
step "5: FLO single-scale (official)"
bash scripts/baseline/run_baseline.sh single FLO ViT-B/32 > logs/single_FLO.log 2>&1
done_ "5"

# ---- [6] FLO ms（batch 版） ----
step "6: FLO multi-scale (batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset FLO --backbone ViT-B/32 \
    --mode real --batch 128 --num_workers 0 > logs/ms_FLO_batch.log 2>&1
done_ "6"

# ---- [7] FLO 生成 ----
step "7: FLO generation"
"$PY" src/analysis/sd_gen_fast.py --dataset FLO --ngen 10 \
    --gen_root_path "$GEN_ROOT" > logs/gen_FLO.log 2>&1
done_ "7"

# ---- [8] FLO 生成图多尺度特征 ----
step "8: FLO gen multi-scale features (batched)"
"$PY" src/analysis/extract_ms_batched.py --dataset FLO --backbone ViT-B/32 \
    --mode gen --batch 128 --gen_root_path "$GEN_ROOT" --image_root "$PROJECT_ROOT/data" \
    > logs/extract_gen_FLO.log 2>&1
done_ "8"

echo "[gpu-queue] ALL DONE"
