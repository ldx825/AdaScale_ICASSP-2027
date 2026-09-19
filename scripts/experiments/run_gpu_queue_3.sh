#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_3.sh — 收官 GPU 队列（在 bench_v2 完成后自动运行）
#   [1] EUROSAT Single 模式 runtime bench（只跑最大 scale 的真实时钟）
#   [2] PET seed replication 特征提取（seed=1234, out_tag=_seed1234）
#   [3] PET ViT-B/16 多尺度特征提取（等权重下载完成）
# 用法: nohup bash scripts/experiments/run_gpu_queue_3.sh > logs/gpu_queue_3.log 2>&1 &
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

step() { echo "[gpu3 $(date +%H:%M:%S)] START $1"; }
done_() { echo "[gpu3 $(date +%H:%M:%S)] DONE  $1 (exit=$?)"; }

echo "[gpu3 $(date +%H:%M:%S)] waiting for bench_v2..."
for i in $(seq 1 120); do
  [ -f outputs/efficiency/PET_adascale_v2.json ] && break
  sleep 30
done

# ---- [1] EUROSAT Single bench ----
step "1: EUROSAT single-mode bench"
"$PY" src/adascale/runtime_bench.py --dataset EUROSAT --backbone ViT-B/32 --mode single \
    --batch 32 --warmup 1 --repeats 2 \
    --out outputs/efficiency/EUROSAT_single.json > logs/bench_eurosat_single.log 2>&1
done_ "1"

# ---- [2] PET seed replication extract ----
step "2: PET seed=1234 multi-scale extraction"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/32 \
    --mode real --seed 1234 --out_tag _seed1234 \
    > logs/extract_seed1234_PET.log 2>&1
done_ "2"

# ---- [3] PET ViT-B/16 extraction（等权重） ----
echo "[gpu3 $(date +%H:%M:%S)] waiting for ViT-B-16 weights..."
for i in $(seq 1 120); do
  sz=$(stat -c%s .home/.cache/clip/ViT-B-16.pt 2>/dev/null || echo 0)
  [ "$sz" = "350837078" ] && break
  sleep 30
done
step "3: PET ViT-B/16 multi-scale extraction"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/16 \
    --mode real > logs/ms_PET_vitb16.log 2>&1
done_ "3"

step "4: PET ViT-B/16 gen features"
"$PY" src/analysis/extract_ms_batched.py --dataset PET --backbone ViT-B/16 --mode gen \
    --gen_root_path "$PROJECT_ROOT/third_party/LG-CLIP/dataset/SD_gen" \
    --image_root "$PROJECT_ROOT/data" > logs/gen_PET_vitb16.log 2>&1
done_ "4"

echo "[gpu3] ALL DONE"
