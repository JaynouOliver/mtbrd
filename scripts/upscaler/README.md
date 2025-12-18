# PBR Material Image Upscaler

Fine-tune Real-ESRGAN on Topaz-upscaled PBR material images.

**Goal:** Replace Topaz Labs API ($0.10/image) with self-hosted solution (<$0.01/image)

## Directory Structure

```
scripts/upscaler/
  - training/           # All training scripts (one-shot setup)
    - README.md         # Detailed training guide
    - setup_instance.sh # Environment setup
    - download_data.py  # Download from Supabase
    - preprocess.py     # Create training patches
    - run_training.sh   # Start training
    - check_training.sh # Monitor progress
    - benchmark.py      # Evaluate model
    - finetune_topaz_esrgan.yml  # Training config
  - inference.py        # Run inference
  - evaluate_model.py   # Full evaluation suite
  - requirements.txt    # Python dependencies
```

## Quick Start (Lambda H100)

### Option 1: One-Shot Setup

```bash
# 1. SSH to Lambda instance
ssh -i ~/.ssh/lightning_rsa ubuntu@<instance-ip>

# 2. Copy training files from local
scp -i ~/.ssh/lightning_rsa -r scripts/upscaler/training ubuntu@<instance-ip>:~/

# 3. Run setup (creates venv, installs deps, downloads models)
chmod +x ~/training/*.sh
./training/setup_instance.sh

# 4. Configure credentials
cat > ~/.env << 'EOF'
SUPABASE_URL=https://glfevldtqujajsalahxd.supabase.co
SUPABASE_KEY=your_key_here
EOF
source ~/.env && export SUPABASE_KEY

# 5. Download and preprocess data
source ~/upscaler_env/bin/activate
python ~/training/download_data.py --limit 40000
python ~/training/preprocess.py

# 6. Start training
./training/run_training.sh

# 7. Monitor
./training/check_training.sh
```

### Option 2: Docker (Alternative)

```bash
docker run --rm --gpus all \
    -v /home/ubuntu/upscaler/datasets:/data \
    subhro2084/realesrgan-trainer \
    python -u realesrgan/train.py -opt /app/config.yml
```

## Training Results (Dec 2024)

Training with ~18K patches from 20K image pairs:

| Iteration | PSNR | SSIM | Notes |
|-----------|------|------|-------|
| Pretrained (baseline) | 21.45 dB | 0.418 | No finetuning |
| 5,000 | 22.16 dB | 0.501 | +0.71 dB improvement |
| 10,000 | 22.17 dB | 0.507 | Plateau begins |
| 15,000 | 22.18 dB | 0.510 | Marginal gains |
| 20,000 | 22.18 dB | 0.510 | Stopped here |

**Key Finding:** With limited data (18K patches), the model plateaus early. To achieve PSNR > 28 dB and SSIM > 0.85, use 100K+ patches.

## Quality Targets

| Metric | Current | Minimum | Target |
|--------|---------|---------|--------|
| SSIM | 0.51 | >= 0.88 | >= 0.92 |
| PSNR | 22.2 dB | >= 28 dB | >= 30 dB |
| LPIPS | TBD | <= 0.12 | <= 0.08 |

## Cost Analysis

| Item | Cost |
|------|------|
| H100 training (~5 hrs) | ~$12-15 |
| Inference per image | ~$0.001-0.002 |
| **Savings for 100K images** | **~$9,800** |

## Inference

```bash
# Single image
python inference.py -i input.jpg -o output.jpg

# Folder of images
python inference.py -i input_folder/ -o output_folder/

# With custom model
python inference.py -i input.jpg -o output.jpg \
    --model path/to/checkpoint.pth
```

## Checkpoints

Best checkpoint from training: `net_g_20000.pth` (134MB)

Download from instance:
```bash
scp -i ~/.ssh/lightning_rsa \
    ubuntu@<instance>:~/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models/net_g_20000.pth \
    ./models/
```

## Lessons Learned

1. **Data quantity matters**: 18K patches insufficient; need 100K+ for quality gains
2. **Domain gap**: Topaz uses proprietary algorithms; Real-ESRGAN style differs
3. **Diminishing returns**: After ~5K iterations with limited data, improvements plateau
4. **Discriminator mismatch**: Using 4x discriminator with 2x model may cause instability

## Future Improvements

1. Download full 40K images, extract 4-8 crops each (~200K patches)
2. Train for 200K+ iterations
3. Try SwinIR or HAT architectures
4. Add LPIPS loss during training
5. Consider training from scratch for this specific domain
