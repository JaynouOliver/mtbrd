#!/bin/bash
# Quick status check for training progress
# Usage: ./check_training.sh

echo "=== TRAINING STATUS ==="
ps aux | grep train.py | grep -v grep | head -1

echo ""
echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "=== LATEST ITERATION ==="
tail -100 /home/ubuntu/training_full.log | grep -v "libpng warning" | grep "iter:" | tail -1

echo ""
echo "=== RECENT LOSS VALUES ==="
tail -100 /home/ubuntu/training_full.log | grep "l_g_pix:" | tail -3

echo ""
echo "=== VALIDATION METRICS (if available) ==="
if [ -f /home/ubuntu/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/train_*.log ]; then
    tail -200 /home/ubuntu/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/train_*.log 2>/dev/null | grep -E "(psnr|ssim)" | tail -3 || echo "No validation yet (first at iter 5000)"
else
    echo "No validation yet (first at iter 5000)"
fi

