#!/usr/bin/env python3
"""
Preprocess downloaded image pairs for Real-ESRGAN training.

1. Verify image pairs are valid
2. Resize HR to exact 2x of LR
3. Create crops/patches for training
4. Generate meta_info file

Usage:
    python preprocess_data.py
"""

import os
import logging
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "topaz_train"
HR_DIR = DATASET_DIR / "hr"
LR_DIR = DATASET_DIR / "lr"

# Output directories (processed)
HR_PROCESSED_DIR = DATASET_DIR / "hr_processed"
LR_PROCESSED_DIR = DATASET_DIR / "lr_processed"
HR_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LR_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Target sizes
LR_TARGET_SIZE = 256  # We'll crop LR to 256x256
SCALE = 2  # 2x upscaling
HR_TARGET_SIZE = LR_TARGET_SIZE * SCALE  # 512x512


def process_pair(lr_path: Path, hr_path: Path) -> dict:
    """Process a single LR/HR pair."""
    try:
        # Load images
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        # Skip if images are too small
        if lr_w < LR_TARGET_SIZE or lr_h < LR_TARGET_SIZE:
            return {"success": False, "reason": "LR too small"}
        
        # Determine crop size based on actual scale ratio
        actual_scale_w = hr_w / lr_w
        actual_scale_h = hr_h / lr_h
        
        # If scale is close to 2x, we're good
        # If not, we'll adjust
        if 1.8 <= actual_scale_w <= 2.2 and 1.8 <= actual_scale_h <= 2.2:
            # Good scale, proceed
            pass
        elif actual_scale_w > 2.2 or actual_scale_h > 2.2:
            # HR is larger than 2x, we need to downscale HR
            hr_img = hr_img.resize((lr_w * SCALE, lr_h * SCALE), Image.LANCZOS)
        else:
            # HR is smaller than 2x, skip this pair
            return {"success": False, "reason": "HR too small relative to LR"}
        
        # Center crop to target size
        # LR crop
        lr_left = (lr_w - LR_TARGET_SIZE) // 2
        lr_top = (lr_h - LR_TARGET_SIZE) // 2
        lr_cropped = lr_img.crop((
            lr_left, lr_top, 
            lr_left + LR_TARGET_SIZE, 
            lr_top + LR_TARGET_SIZE
        ))
        
        # HR crop (at 2x the position and size)
        hr_left = lr_left * SCALE
        hr_top = lr_top * SCALE
        
        # Ensure HR image is large enough after potential resize
        hr_w_new, hr_h_new = hr_img.size
        if hr_w_new < HR_TARGET_SIZE or hr_h_new < HR_TARGET_SIZE:
            return {"success": False, "reason": "HR too small after resize"}
        
        hr_cropped = hr_img.crop((
            hr_left, hr_top,
            hr_left + HR_TARGET_SIZE,
            hr_top + HR_TARGET_SIZE
        ))
        
        # Verify sizes
        if lr_cropped.size != (LR_TARGET_SIZE, LR_TARGET_SIZE):
            return {"success": False, "reason": "LR crop failed"}
        if hr_cropped.size != (HR_TARGET_SIZE, HR_TARGET_SIZE):
            return {"success": False, "reason": "HR crop failed"}
        
        # Save processed images
        filename = lr_path.stem + ".png"
        lr_out = LR_PROCESSED_DIR / filename
        hr_out = HR_PROCESSED_DIR / filename
        
        lr_cropped.save(lr_out, "PNG")
        hr_cropped.save(hr_out, "PNG")
        
        return {"success": True, "filename": filename}
        
    except Exception as e:
        return {"success": False, "reason": str(e)}


def main():
    logger.info("Starting preprocessing...")
    
    # Get all LR images
    lr_files = list(LR_DIR.glob("*.jpg")) + list(LR_DIR.glob("*.png"))
    logger.info(f"Found {len(lr_files)} LR images")
    
    successful = []
    failed = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for lr_path in lr_files:
            hr_path = HR_DIR / lr_path.name
            if hr_path.exists():
                futures[executor.submit(process_pair, lr_path, hr_path)] = lr_path
        
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                successful.append(result["filename"])
            else:
                failed += 1
            
            total = len(successful) + failed
            if total % 500 == 0:
                logger.info(f"Processed: {total}/{len(futures)}")
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {failed}")
    
    # Generate meta_info file for BasicSR
    meta_file = DATASET_DIR / "meta_info_topaz.txt"
    with open(meta_file, 'w') as f:
        for filename in sorted(successful):
            # Format: hr_filename, lr_filename (for paired dataset)
            f.write(f"{filename}\n")
    
    logger.info(f"Meta info saved to {meta_file}")
    
    # Create train/val split
    train_count = int(len(successful) * 0.95)
    train_files = successful[:train_count]
    val_files = successful[train_count:]
    
    train_meta = DATASET_DIR / "meta_info_topaz_train.txt"
    val_meta = DATASET_DIR / "meta_info_topaz_val.txt"
    
    with open(train_meta, 'w') as f:
        for filename in train_files:
            f.write(f"{filename}\n")
    
    with open(val_meta, 'w') as f:
        for filename in val_files:
            f.write(f"{filename}\n")
    
    logger.info(f"Train set: {len(train_files)} images")
    logger.info(f"Val set: {len(val_files)} images")


if __name__ == "__main__":
    main()


