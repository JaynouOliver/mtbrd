#!/usr/bin/env python3
"""
Preprocess downloaded images into training patches.
Creates 256x256 LR patches and 512x512 HR patches for 2x upscaling.

Usage:
    python preprocess.py
"""
import logging
import random
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess')

BASE = Path('/home/ubuntu/upscaler/datasets/topaz_train')
HR_IN, LR_IN = BASE / 'hr', BASE / 'lr'
HR_OUT, LR_OUT = BASE / 'hr_processed', BASE / 'lr_processed'
HR_OUT.mkdir(parents=True, exist_ok=True)
LR_OUT.mkdir(parents=True, exist_ok=True)

LR_SIZE = 256
SCALE = 2
HR_SIZE = LR_SIZE * SCALE
N_CROPS = 1  # Single crop per image (increase to 4-8 for more patches)


def process_pair(stem):
    """Process a single image pair into patches."""
    lr_path = LR_IN / f'{stem}.jpg'
    hr_path = HR_IN / f'{stem}.jpg'
    
    if not lr_path.exists():
        lr_path = LR_IN / f'{stem}.png'
    if not hr_path.exists():
        hr_path = HR_IN / f'{stem}.png'
    
    if not (lr_path.exists() and hr_path.exists()):
        return []
    
    try:
        lr = Image.open(lr_path).convert('RGB')
        hr = Image.open(hr_path).convert('RGB')
    except Exception:
        return []
    
    lw, lh = lr.size
    hw, hh = hr.size
    
    if lw == 0 or lh == 0:
        return []
    
    # Check scale ratio
    sw, sh = hw / lw, hh / lh
    if sw < 1.5 or sw > 3.0 or sh < 1.5 or sh > 3.0:
        return []
    
    if lw < LR_SIZE or lh < LR_SIZE:
        return []
    
    # Resize HR to exact 2x of LR
    hr = hr.resize((lw * SCALE, lh * SCALE), Image.LANCZOS)
    
    outputs = []
    for idx in range(N_CROPS):
        if lw > LR_SIZE:
            left = random.randint(0, lw - LR_SIZE)
        else:
            left = 0
        if lh > LR_SIZE:
            top = random.randint(0, lh - LR_SIZE)
        else:
            top = 0
        
        lr_crop = lr.crop((left, top, left + LR_SIZE, top + LR_SIZE))
        hr_crop = hr.crop((left * SCALE, top * SCALE, left * SCALE + HR_SIZE, top * SCALE + HR_SIZE))
        
        name = f'{stem}_{idx:02d}.png'
        lr_crop.save(LR_OUT / name, 'PNG')
        hr_crop.save(HR_OUT / name, 'PNG')
        outputs.append(name)
    
    return outputs


if __name__ == '__main__':
    stems = [p.stem for p in LR_IN.glob('*.jpg')] + [p.stem for p in LR_IN.glob('*.png')]
    stems = list(sorted(set(stems)))
    logger.info(f'Found {len(stems)} image pairs to process')
    
    all_names = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_pair, stem): stem for stem in stems}
        for fut in as_completed(futures):
            names = fut.result()
            all_names.extend(names)
            processed += 1
            if processed % 2000 == 0:
                logger.info(f'Processed {processed}/{len(stems)} pairs, {len(all_names)} patches')
    
    # Train/val split (95/5)
    random.shuffle(all_names)
    split_idx = int(len(all_names) * 0.95)
    train_names = all_names[:split_idx]
    val_names = all_names[split_idx:]
    
    # Write meta files
    with open(BASE / 'meta_info_train.txt', 'w') as f:
        for n in train_names:
            f.write(f'{n}, {n}\n')
    
    with open(BASE / 'meta_info_val.txt', 'w') as f:
        for n in val_names:
            f.write(f'{n}, {n}\n')
    
    logger.info(f'Preprocessing complete:')
    logger.info(f'  Total patches: {len(all_names)}')
    logger.info(f'  Train: {len(train_names)}')
    logger.info(f'  Val: {len(val_names)}')

