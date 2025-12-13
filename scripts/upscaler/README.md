# Image Upscaler Training

Fine-tune Real-ESRGAN on Topaz-upscaled PBR material images.

## Quick Start (Lambda Labs)

```bash
# 1. SSH into your Lambda instance
ssh ubuntu@<your-lambda-ip>

# 2. Clone this repo
git clone <your-repo-url>
cd mtbrd/scripts/upscaler

# 3. Set Supabase key
export SUPABASE_KEY="your-supabase-anon-key"

# 4. Run everything
chmod +x setup_and_train.sh
./setup_and_train.sh
```

## Manual Step-by-Step

```bash
# Install dependencies
pip install -r requirements.txt

# Download training data (10k images)
python download_training_data.py --limit 10000 --workers 50

# Preprocess data
python preprocess_data.py

# Start training
cd Real-ESRGAN
python realesrgan/train.py -opt options/finetune_realesrgan_x2_fast.yml --auto_resume
```

## Test Inference

```bash
python test_inference.py -i test_images/ -o results/
```

## Files

| File | Description |
|------|-------------|
| `setup_and_train.sh` | One-command setup and training |
| `download_training_data.py` | Downloads LR/HR pairs from Supabase |
| `preprocess_data.py` | Prepares data for training |
| `finetune_realesrgan_x2_fast.yml` | Training config (50k iterations) |
| `test_inference.py` | Test the trained model |
| `FAST_TRACK_PLAN.md` | Detailed plan and timeline |

## Expected Timeline

| Phase | Duration |
|-------|----------|
| Setup | 15 min |
| Download 10k images | 1-1.5 hrs |
| Preprocess | 30 min |
| Training (1x A100) | 3-4 hrs |
| Training (4x A100) | 1-1.5 hrs |
| **Total** | **4-6 hrs** |

## Cost

- Lambda A100 80GB: $1.79/hr
- Total training cost: ~$10-20


