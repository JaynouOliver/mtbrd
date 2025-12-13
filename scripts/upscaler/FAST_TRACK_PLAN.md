# Fast-Track Image Upscaler Training Plan

## Goal: Fine-tuned model with inference in 4-6 hours

---

## Timeline Overview

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Setup Lambda instance | 15 min | 0:15 |
| 2. Download 10k image pairs | 1-1.5 hrs | 1:45 |
| 3. Preprocess data | 30 min | 2:15 |
| 4. Fine-tune model | 2-3 hrs | 5:15 |
| 5. Test inference | 15 min | 5:30 |

---

## Phase 1: Lambda Instance Setup (15 min)

### Recommended Instance
- **GPU**: 1x NVIDIA A100 80GB ($1.79/hr) or 4x A100 for faster training
- **Storage**: Use included SSD (plenty of space)

### Initial Setup Commands
```bash
# SSH into Lambda instance
ssh ubuntu@<your-lambda-ip>

# Clone this repo
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# Install dependencies
pip install basicsr realesrgan
pip install -r requirements.txt

# Download pretrained model (we'll fine-tune from this)
mkdir -p experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
  -O experiments/pretrained_models/RealESRGAN_x2plus.pth

# Also download the x4 model as backup
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O experiments/pretrained_models/RealESRGAN_x4plus.pth
```

---

## Phase 2: Download Training Data (1-1.5 hrs)

### Strategy: 10k high-quality pairs
- Recent images (since 2025-02-01) - likely higher quality
- Skip pairs where original == upscaled
- Download in parallel (50 workers)

### Run the download script
```bash
python download_training_data.py --limit 10000 --workers 50
```

Expected output structure:
```
datasets/
├── topaz_train/
│   ├── hr/  (upscaled by Topaz - ground truth)
│   └── lr/  (original low-res)
└── meta_info_topaz.txt
```

---

## Phase 3: Preprocess Data (30 min)

### What this does:
1. Verify image pairs are valid
2. Resize HR images to exact 2x of LR
3. Create patches (256x256) for training
4. Generate meta_info file

### Run preprocessing
```bash
python preprocess_data.py
```

---

## Phase 4: Fine-tune Model (2-3 hrs)

### Fast Training Config
- **Iterations**: 50,000 (instead of 400k)
- **Batch size**: 16 per GPU
- **Learning rate**: 2e-4 (aggressive for fast convergence)
- **Validation**: Every 5k iterations

### Single GPU Training
```bash
python realesrgan/train.py \
  -opt options/finetune_realesrgan_x2_fast.yml \
  --auto_resume
```

### Multi-GPU Training (4x faster)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
  realesrgan/train.py \
  -opt options/finetune_realesrgan_x2_fast.yml \
  --launcher pytorch --auto_resume
```

### Expected Training Speed
- 1x A100: ~500 iter/min = 100 min for 50k
- 4x A100: ~2000 iter/min = 25 min for 50k

---

## Phase 5: Test Inference (15 min)

### Quick test
```bash
python inference_realesrgan.py \
  -n RealESRGAN_x2plus_Topaz \
  -i test_images/ \
  -o results/ \
  -s 2
```

### Verify output
- Check results/ folder for upscaled images
- Compare visually with Topaz outputs

---

## Files to Create

1. `download_training_data.py` - Downloads image pairs from Supabase
2. `preprocess_data.py` - Prepares data for training
3. `finetune_realesrgan_x2_fast.yml` - Training config
4. `test_inference.py` - Quick inference test

---

## Quick Decisions Made for Speed

| Decision | Fast Choice | Why |
|----------|-------------|-----|
| Training images | 10k (not 40k) | 4x faster download, 4x faster training |
| Iterations | 50k (not 400k) | 8x faster, still good for fine-tuning |
| Two-phase training | Skip, just GAN | Saves 3-4 hours |
| Validation | Every 5k iter | Reduced overhead |
| Data augmentation | Basic (flip/rot) | Standard, no extra time |

---

## Cost Estimate

| Resource | Hours | Cost |
|----------|-------|------|
| 1x A100 80GB | 5 hrs | $9 |
| 4x A100 80GB | 3 hrs | $21 |

**Total: $9-21**

---

## Success Criteria for Today

1. Model trains without errors
2. Produces upscaled output images
3. Visual quality looks reasonable (not perfect yet)

Benchmarking and optimization can happen tomorrow.


