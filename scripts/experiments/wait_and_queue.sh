#!/usr/bin/env bash
# 等待 PET 生成完成（370 张），然后自动启动 GPU 队列 1
cd /root/autodl-tmp/EviZO-VP/AdaScale-CLIP
for i in $(seq 1 240); do
  N=$(find third_party/LG-CLIP/dataset/SD_gen/SD_2.1_PET_10 -name "*.jpg" 2>/dev/null | wc -l)
  if [ "$N" -ge 370 ]; then
    echo "[wait] PET gen complete: $N images at $(date +%H:%M:%S)"
    break
  fi
  sleep 30
done
if [ "$N" -lt 370 ]; then echo "[wait] TIMEOUT at $N images"; exit 1; fi
sleep 5
bash scripts/experiments/run_gpu_queue_1.sh
