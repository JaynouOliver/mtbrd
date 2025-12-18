#!/bin/bash
# One-shot setup script for Lambda H100 instance
# Usage: ./setup_instance.sh
#
# Prerequisites:
#   - SSH access to Lambda instance
#   - .env file with SUPABASE_KEY set
#
# This script:
#   1. Creates virtual environment
#   2. Installs dependencies
#   3. Clones Real-ESRGAN
#   4. Downloads pretrained models

set -e
LOG=~/setup.log
exec > >(tee -a $LOG) 2>&1

echo "=== Setup started at $(date) ==="

# Create and activate virtual environment
if [ ! -d ~/upscaler_env ]; then
    python3 -m venv ~/upscaler_env
fi
source ~/upscaler_env/bin/activate
echo "Python: $(which python)"

# System dependencies
sudo apt update -qq
sudo apt install -y cython3 build-essential python3-dev

# Install Python packages
echo "Installing numpy..."
pip install 'numpy<2' --quiet

echo "Installing PyTorch (should be pre-installed)..."
pip install torch torchvision --quiet || true

echo "Installing opencv..."
pip install opencv-python --quiet

echo "Installing basicsr..."
BASICSR_EXT=False pip install basicsr --no-build-isolation --quiet

echo "Installing facexlib gfpgan..."
pip install facexlib gfpgan --quiet

echo "Installing realesrgan..."
pip install realesrgan --quiet

echo "Installing other dependencies..."
pip install requests tqdm lmdb lpips scikit-image --quiet

# Clone Real-ESRGAN
if [ ! -d ~/Real-ESRGAN ]; then
    echo "Cloning Real-ESRGAN..."
    cd ~
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    pip install -e . --quiet
else
    echo "Real-ESRGAN already exists"
fi

# Download pretrained models
echo "Downloading pretrained models..."
mkdir -p ~/Real-ESRGAN/experiments/pretrained_models

if [ ! -f ~/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x2plus.pth ]; then
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
        -O ~/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x2plus.pth
fi

if [ ! -f ~/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus_netD.pth ]; then
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_netD.pth \
        -O ~/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus_netD.pth
fi

# Create data directories
mkdir -p ~/upscaler/datasets/topaz_train/{hr,lr,hr_processed,lr_processed}

echo ""
echo "=== Setup complete at $(date) ==="
echo "Packages installed:"
pip list | grep -E "torch|basicsr|realesrgan|numpy|opencv"

echo ""
echo "Next steps:"
echo "  1. Set SUPABASE_KEY in ~/.env"
echo "  2. Run: python download_data.py --limit 20000"
echo "  3. Run: python preprocess.py"
echo "  4. Start training with run_training.sh"

