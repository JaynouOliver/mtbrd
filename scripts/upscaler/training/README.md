# Real-ESRGAN Training for Topaz-Quality Upscaling

This directory contains all scripts needed to train a Real-ESRGAN model to replicate Topaz upscaling quality.

## Quick Start (One-Shot Setup)

### 1. SSH into Lambda H100 Instance

```bash
ssh -i ~/.ssh/lightning_rsa ubuntu@<instance-ip>
```

### 2. Copy Training Files

```bash
# From local machine
scp -i ~/.ssh/lightning_rsa -r scripts/upscaler/training ubuntu@<instance-ip>:~/
```

### 3. Setup Environment

```bash
chmod +x ~/training/*.sh
./training/setup_instance.sh
```

### 4. Configure Environment Variables

Create `~/.env`:
```bash
cat > ~/.env << 'EOF'
SUPABASE_URL=https://glfevldtqujajsalahxd.supabase.co
SUPABASE_KEY=your_anon_key_here
EOF
source ~/.env
export SUPABASE_KEY
```

### 5. Download Data

```bash
source ~/upscaler_env/bin/activate
python ~/training/download_data.py --limit 40000 --workers 64
```

### 6. Preprocess Data

```bash
python ~/training/preprocess.py
```

### 7. Start Training

```bash
./training/run_training.sh
```

### 8. Monitor Training

```bash
./training/check_training.sh
# or
tail -f ~/training_full.log | grep -v "libpng warning"
```

## Files

| File | Description |
|------|-------------|
| `setup_instance.sh` | One-shot environment setup |
| `download_data.py` | Downloads images from Supabase |
| `preprocess.py` | Creates training patches |
| `finetune_topaz_esrgan.yml` | Training configuration |
| `run_training.sh` | Starts training |
| `check_training.sh` | Quick status check |
| `benchmark.py` | Evaluates model quality |

## Training Configuration

Key parameters in `finetune_topaz_esrgan.yml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `scale` | 2 | 2x upscaling |
| `batch_size_per_gpu` | 12 | Adjust for GPU memory |
| `total_iter` | 100000 | ~9 hours on H100 |
| `gt_size` | 256 | HR patch size |
| `val_freq` | 5000 | Validation interval |
| `save_checkpoint_freq` | 10000 | Checkpoint interval |

## Expected Metrics

Based on training with ~18K patches:

| Iteration | PSNR | SSIM |
|-----------|------|------|
| 5,000 | ~22.2 dB | ~0.50 |
| 10,000 | ~22.2 dB | ~0.51 |
| 20,000 | ~22.2 dB | ~0.51 |

**Note**: To achieve higher quality (PSNR > 28, SSIM > 0.85):
- Use 40K+ images with 4-8 crops each (~160K-320K patches)
- Train for 200K+ iterations
- Consider training from scratch instead of finetuning

## Docker Alternative

If you prefer Docker:

```bash
docker run --rm --gpus all \
    -v /home/ubuntu/upscaler/datasets:/data \
    -v /home/ubuntu/Real-ESRGAN:/app/Real-ESRGAN \
    subhro2084/realesrgan-trainer \
    python -u realesrgan/train.py -opt options/finetune_topaz_esrgan.yml
```

## Benchmarking

After training:

```bash
source ~/upscaler_env/bin/activate
python ~/training/benchmark.py \
    --model ~/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models/net_g_20000.pth \
    --samples 100
```

## Checkpoints

Checkpoints are saved to:
```
~/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models/
  - net_g_10000.pth  # Generator at 10K iters
  - net_g_20000.pth  # Generator at 20K iters
  - net_d_*.pth      # Discriminator checkpoints
```

To download best checkpoint:
```bash
scp -i ~/.ssh/lightning_rsa ubuntu@<instance-ip>:~/Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models/net_g_20000.pth ./
```

## Cost Estimate

| Instance | Cost/Hour | Training Time | Total Cost |
|----------|-----------|---------------|------------|
| H100 | ~$2.50 | 9 hours (100K iters) | ~$22.50 |
| H100 | ~$2.50 | 18 hours (200K iters) | ~$45.00 |

## Lessons Learned

1. **Data is key**: 18K patches is insufficient for matching Topaz quality. Need 100K+ patches.
2. **Domain gap**: Real-ESRGAN pretrained on natural images, Topaz uses different enhancement style.
3. **Diminishing returns**: After ~5K iterations, improvements slow significantly with limited data.
4. **Discriminator mismatch**: Using 4x discriminator with 2x model may cause issues.

## Future Improvements

1. Train from scratch with 200K+ image pairs
2. Try SwinIR or HAT architectures
3. Adjust loss weights for Topaz-specific style
4. Use LPIPS loss during training

