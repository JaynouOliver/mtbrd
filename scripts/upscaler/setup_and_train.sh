#!/bin/bash
# =============================================================================
# FAST-TRACK SETUP AND TRAINING SCRIPT
# Complete setup and training in one go
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Real-ESRGAN Fast-Track Training Setup"
echo "=============================================="

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for SUPABASE_KEY
if [ -z "$SUPABASE_KEY" ]; then
    echo "ERROR: SUPABASE_KEY environment variable not set"
    echo "Set it with: export SUPABASE_KEY='your-key-here'"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "[1/6] Installing dependencies..."
echo "=============================================="

pip install --quiet basicsr realesrgan requests pillow

# Clone Real-ESRGAN if not exists
if [ ! -d "Real-ESRGAN" ]; then
    echo "Cloning Real-ESRGAN repository..."
    git clone https://github.com/xinntao/Real-ESRGAN.git
fi

cd Real-ESRGAN
pip install --quiet -r requirements.txt
pip install --quiet -e .
cd ..

# Step 2: Download pretrained model
echo ""
echo "[2/6] Downloading pretrained model..."
echo "=============================================="

mkdir -p experiments/pretrained_models
if [ ! -f "experiments/pretrained_models/RealESRGAN_x2plus.pth" ]; then
    wget -q --show-progress \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
        -O experiments/pretrained_models/RealESRGAN_x2plus.pth
    echo "Downloaded RealESRGAN_x2plus.pth"
else
    echo "Pretrained model already exists"
fi

# Step 3: Download training data
echo ""
echo "[3/6] Downloading training data (10k images)..."
echo "=============================================="

python download_training_data.py --limit 10000 --workers 50

# Step 4: Preprocess data
echo ""
echo "[4/6] Preprocessing training data..."
echo "=============================================="

python preprocess_data.py

# Step 5: Copy training config
echo ""
echo "[5/6] Setting up training configuration..."
echo "=============================================="

# Copy config to Real-ESRGAN options folder
mkdir -p Real-ESRGAN/options
cp finetune_realesrgan_x2_fast.yml Real-ESRGAN/options/

# Create symlinks for data access
cd Real-ESRGAN
ln -sf ../datasets datasets 2>/dev/null || true
ln -sf ../experiments experiments 2>/dev/null || true
cd ..

# Step 6: Start training
echo ""
echo "[6/6] Starting training..."
echo "=============================================="
echo ""
echo "Training with 50k iterations..."
echo "Expected time: 2-3 hours on 4x A100, 4-5 hours on 1x A100"
echo ""
echo "Monitor progress in: experiments/RealESRGAN_x2plus_Topaz_fast/"
echo ""

# Detect GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

cd Real-ESRGAN

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Using distributed training on $GPU_COUNT GPUs..."
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1))) \
    python -m torch.distributed.launch \
        --nproc_per_node=$GPU_COUNT \
        --master_port=4321 \
        realesrgan/train.py \
        -opt options/finetune_realesrgan_x2_fast.yml \
        --launcher pytorch \
        --auto_resume
else
    echo "Using single GPU training..."
    python realesrgan/train.py \
        -opt options/finetune_realesrgan_x2_fast.yml \
        --auto_resume
fi

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Model saved to: experiments/RealESRGAN_x2plus_Topaz_fast/models/"
echo ""
echo "To test inference:"
echo "  cd $SCRIPT_DIR"
echo "  python test_inference.py -i test_images/ -o results/"
echo ""


