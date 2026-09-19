#!/usr/bin/env bash
# =============================================================================
# run_cpu_queue_6.sh — 分析链（等 GPU 队列全部完成后自动运行）
#   [1] PET seed=1234 复现分析（text）
#   [2] PET ViT-B/16 audit + sim + AdaScale + bootstrap（text + gen）
#   [3] 汇总输出
# 用法: nohup bash scripts/experiments/run_cpu_queue_6.sh > logs/cpu_queue_6.log 2>&1 &
# =============================================================================
set -uo pipefail
PROJECT_ROOT=/root/autodl-tmp/EviZO-VP/AdaScale-CLIP
cd "$PROJECT_ROOT"
export HOME="$PROJECT_ROOT/.home"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
PY="$PROJECT_ROOT/.venv/bin/python"

step() { echo "[cpu6 $(date +%H:%M:%S)] START $1"; }
done_() { echo "[cpu6 $(date +%H:%M:%S)] DONE  $1 (exit=$?)"; }

echo "[cpu6 $(date +%H:%M:%S)] waiting for GPU queues (queue3 + queue4 + queue5)..."
for i in $(seq 1 360); do
  if grep -q "\[gpu3\] ALL DONE" logs/gpu_queue_3.log 2>/dev/null && \
     grep -q "\[gpu4\] ALL DONE" logs/gpu_queue_4.log 2>/dev/null && \
     grep -q "DONE" logs/queue5.log 2>/dev/null; then
    break
  fi
  sleep 60
done
echo "[cpu6 $(date +%H:%M:%S)] all GPU queues done"

# ---- [1] seed replication (text) ----
step "1: seed=1234 replication analysis (PET/text)"
"$PY" scripts/final/seed_replication.py --dataset PET --proto text --tag _seed1234
done_ "1"

# ---- [2] ViT-B/16 (PET, text + gen) ----
for proto in text gen; do
  step "2: ViT-B/16 audit ($proto)"
  "$PY" scripts/audit/run_scale_audit.py --dataset PET --backbone ViT-B/16 --proto $proto
  done_ "2a-$proto"
  step "2: ViT-B/16 simulation ($proto)"
  "$PY" scripts/experiments/exp_simulation.py --dataset PET --backbone ViT-B/16 --proto $proto
  done_ "2b-$proto"
  step "2: ViT-B/16 AdaScale final ($proto)"
  "$PY" scripts/final/evaluate_adascale_final.py --dataset PET --backbone ViT-B/16 \
      --proto $proto --stab 4,0.45,0.01
  done_ "2c-$proto"
  step "2: ViT-B/16 bootstrap ($proto)"
  "$PY" scripts/final/bootstrap_v2.py --dataset PET --backbone ViT-B/16 --proto $proto
  done_ "2d-$proto"
done

echo "[cpu6 $(date +%H:%M:%S)] ALL DONE"
