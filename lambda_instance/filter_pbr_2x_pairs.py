import glob
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SRC_ORIG_DIR = Path("downloads/materials/original")
SRC_UPSCALED_DIR = Path("downloads/materials/upscaled")
DST_ROOT = Path("datasets/pbr_2x")

# Train/val split ratio (95% train, 5% val)
TRAIN_SPLIT = 0.95

# Tolerance around exact 2x
MIN_SCALE = 1.9
MAX_SCALE = 2.1
RESIZE_GT_TO_EXACT_2X = True
NUM_WORKERS = 8  # parallel workers


def process_pair(args):
    """Process a single LQ/GT pair and return status."""
    lq_path, gt_path, dst_lq, dst_gt, output_name = args

    if not gt_path.exists():
        return ("skipped", output_name, "GT file missing")

    try:
        with Image.open(lq_path) as im_lq, Image.open(gt_path) as im_gt:
            w_lq, h_lq = im_lq.size
            w_gt, h_gt = im_gt.size

        if w_lq == 0 or h_lq == 0:
            return ("skipped", output_name, "Invalid LQ dimensions")

        sx = w_gt / w_lq
        sy = h_gt / h_lq

        # require roughly uniform ~2x scaling
        if not (MIN_SCALE <= sx <= MAX_SCALE and MIN_SCALE <= sy <= MAX_SCALE):
            return ("skipped", output_name, f"Scale mismatch: sx={sx:.2f}, sy={sy:.2f}")

        out_lq = dst_lq / output_name
        out_gt = dst_gt / output_name

        shutil.copy2(lq_path, out_lq)

        if RESIZE_GT_TO_EXACT_2X:
            target_size = (w_lq * 2, h_lq * 2)
            with Image.open(gt_path) as im_gt:
                im_gt = im_gt.resize(target_size, Image.BICUBIC)
                im_gt.save(out_gt)
        else:
            shutil.copy2(gt_path, out_gt)

        return ("kept", output_name, f"LQ={w_lq}x{h_lq}, GT={w_lq*2}x{h_lq*2}")
    except Exception as e:
        return ("error", output_name, str(e))


def match_pairs():
    """Match original and upscaled images by their index number."""
    import re
    
    orig_files = sorted(glob.glob(str(SRC_ORIG_DIR / "*")))
    upscaled_files = sorted(glob.glob(str(SRC_UPSCALED_DIR / "*")))
    
    logger.info(f"Found {len(orig_files)} original images")
    logger.info(f"Found {len(upscaled_files)} upscaled images")
    
    # Extract index from filename (e.g., "00001_orig.jpg" -> "00001")
    def extract_index(filename):
        # Match leading digits
        match = re.match(r"^(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else None
    
    # Create mappings by index
    orig_by_idx = {}
    for orig_path in orig_files:
        idx = extract_index(orig_path)
        if idx is not None:
            orig_by_idx[idx] = Path(orig_path)
    
    upscaled_by_idx = {}
    for upscaled_path in upscaled_files:
        idx = extract_index(upscaled_path)
        if idx is not None:
            upscaled_by_idx[idx] = Path(upscaled_path)
    
    # Match pairs by index
    pairs = []
    for idx in sorted(set(orig_by_idx.keys()) & set(upscaled_by_idx.keys())):
        pairs.append((orig_by_idx[idx], upscaled_by_idx[idx]))
    
    logger.info(f"Matched {len(pairs)} pairs")
    return pairs


def process_split(pairs_list, split: str, start_idx: int, end_idx: int) -> None:
    """Process train or val split with parallel workers."""
    logger.info(f"Starting {split} split processing...")
    logger.info(f"Processing pairs {start_idx} to {end_idx}")

    dst_lq = DST_ROOT / split / "LQ"
    dst_gt = DST_ROOT / split / "GT"

    dst_lq.mkdir(parents=True, exist_ok=True)
    dst_gt.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories: {dst_lq}, {dst_gt}")

    split_pairs = pairs_list[start_idx:end_idx]
    total = len(split_pairs)
    logger.info(f"Processing {total} pairs for {split} split")

    if total == 0:
        logger.warning(f"No pairs to process for {split}")
        return

    # Prepare arguments for parallel processing
    # Output names are sequential: 000001.png, 000002.png, etc.
    args_list = []
    for idx, (lq_path, gt_path) in enumerate(split_pairs, start=1):
        output_name = f"{idx:06d}.png"
        args_list.append((lq_path, gt_path, dst_lq, dst_gt, output_name))

    kept = 0
    skipped = 0
    errors = 0
    processed = 0

    logger.info(f"Starting parallel processing with {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_pair, args): args for args in args_list}

        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result[0] == "kept":
                kept += 1
            elif result[0] == "skipped":
                skipped += 1
            else:
                errors += 1
                logger.warning(f"Error processing {result[1]}: {result[2]}")

            # Progress update every 500 files
            if processed % 500 == 0:
                logger.info(
                    f"[{split}] Progress: {processed}/{total} "
                    f"(kept: {kept}, skipped: {skipped}, errors: {errors})"
                )

    logger.info(
        f"[{split}] COMPLETE: kept {kept} pairs, skipped {skipped}, errors {errors} "
        f"(total processed: {processed})"
    )


def main() -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Starting PBR 2x dataset filtering (direct from downloads/materials)")
    logger.info(f"Source original: {SRC_ORIG_DIR}")
    logger.info(f"Source upscaled: {SRC_UPSCALED_DIR}")
    logger.info(f"Destination: {DST_ROOT}")
    logger.info(f"Train/val split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%")
    logger.info(f"Scale tolerance: {MIN_SCALE} - {MAX_SCALE}")
    logger.info(f"Resize GT to exact 2x: {RESIZE_GT_TO_EXACT_2X}")
    logger.info(f"Parallel workers: {NUM_WORKERS}")
    logger.info("=" * 60)

    # Check source directories exist
    if not SRC_ORIG_DIR.exists():
        logger.error(f"Source original directory does not exist: {SRC_ORIG_DIR}")
        return
    if not SRC_UPSCALED_DIR.exists():
        logger.error(f"Source upscaled directory does not exist: {SRC_UPSCALED_DIR}")
        return

    # Match pairs from downloads/materials
    logger.info("Matching original and upscaled image pairs...")
    pairs = match_pairs()
    
    if len(pairs) == 0:
        logger.error("No matching pairs found!")
        return

    # Apply train/val split
    total_pairs = len(pairs)
    train_count = int(total_pairs * TRAIN_SPLIT)
    val_count = total_pairs - train_count
    
    logger.info(f"Split: {train_count} pairs for train, {val_count} pairs for val")
    logger.info("-" * 60)

    # Process train split
    process_split(pairs, "train", 0, train_count)
    logger.info("-" * 60)

    # Process val split
    process_split(pairs, "val", train_count, total_pairs)
    logger.info("-" * 60)

    logger.info("All splits processed successfully!")
    logger.info(f"Clean 2x dataset available at: {DST_ROOT}")


if __name__ == "__main__":
    main()