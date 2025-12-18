#!/bin/bash
# Start Real-ESRGAN training
# Usage: ./run_training.sh [--resume]
#
# Logs to ~/training_full.log

set -e

source ~/upscaler_env/bin/activate

# Copy config file
cp ~/training/finetune_topaz_esrgan.yml ~/Real-ESRGAN/options/

cd ~/Real-ESRGAN

# Check for resume flag
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--auto_resume"
    echo "Resuming from latest checkpoint..."
fi

echo "Starting training at $(date)"
echo "=============================================="
echo "Config: options/finetune_topaz_esrgan.yml"
echo "Log: ~/training_full.log"
echo "=============================================="

# Run training with output to log file
nohup python -u realesrgan/train.py \
    -opt options/finetune_topaz_esrgan.yml \
    $RESUME_FLAG \
    > ~/training_full.log 2>&1 &

echo "Training started in background (PID: $!)"
echo ""
echo "Monitor with:"
echo "  tail -f ~/training_full.log | grep -v 'libpng warning'"
echo "  ./check_training.sh"

