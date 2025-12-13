#!/usr/bin/env python3
"""
Fast download of training image pairs from Supabase.
Downloads original (LR) and Topaz-upscaled (HR) images.

Usage:
    python download_training_data.py --limit 10000 --workers 50
"""

import os
import sys
import json
import logging
import argparse
import requests
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://glfevldtqujajsalahxd.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Output directories
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "topaz_train"
HR_DIR = DATASET_DIR / "hr"  # Topaz upscaled (ground truth)
LR_DIR = DATASET_DIR / "lr"  # Original (low-res)

# Create directories
HR_DIR.mkdir(parents=True, exist_ok=True)
LR_DIR.mkdir(parents=True, exist_ok=True)


def fetch_image_pairs(limit: int = 10000, offset: int = 0) -> list:
    """Fetch image pairs from Supabase."""
    
    if not SUPABASE_KEY:
        logger.error("Set SUPABASE_KEY environment variable")
        sys.exit(1)
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    # Query for valid pairs where original != upscaled
    # Using RPC for complex query
    query = f"""
    SELECT 
        id,
        metadata->>'materialImageUrl' AS original_url,
        "materialData"->'files'->>'color_original' AS upscaled_url
    FROM public."productsV2"
    WHERE 
        "productType" IN ('fixed material', 'material')
        AND "objectStatus" IN ('APPROVED', 'APPROVED_PRO')
        AND metadata->>'materialImageUrl' IS NOT NULL
        AND "materialData"->'files'->>'color_original' IS NOT NULL
        AND metadata->>'materialImageUrl' != "materialData"->'files'->>'color_original'
        AND to_timestamp("updatedAt" / 1000) >= '2025-02-01'
    ORDER BY "updatedAt" DESC
    LIMIT {limit}
    OFFSET {offset}
    """
    
    # Use REST API with filter
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_training_pairs"
    
    # Alternative: direct table query with filters
    # For speed, we'll paginate through the table
    all_pairs = []
    page_size = 1000
    current_offset = offset
    
    while len(all_pairs) < limit:
        remaining = limit - len(all_pairs)
        fetch_size = min(page_size, remaining)
        
        # Build query URL
        query_url = (
            f"{SUPABASE_URL}/rest/v1/productsV2"
            f"?select=id,metadata,materialData"
            f"&productType=in.(material,fixed%20material)"
            f"&objectStatus=in.(APPROVED,APPROVED_PRO)"
            f"&order=updatedAt.desc"
            f"&limit={fetch_size}"
            f"&offset={current_offset}"
        )
        
        try:
            response = requests.get(query_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            # Extract valid pairs
            for item in data:
                metadata = item.get("metadata", {}) or {}
                material_data = item.get("materialData", {}) or {}
                files = material_data.get("files", {}) or {}
                
                original_url = metadata.get("materialImageUrl")
                upscaled_url = files.get("color_original")
                
                # Skip invalid pairs
                if not original_url or not upscaled_url:
                    continue
                if original_url == upscaled_url:
                    continue
                
                all_pairs.append({
                    "id": item["id"],
                    "original_url": original_url,
                    "upscaled_url": upscaled_url
                })
                
                if len(all_pairs) >= limit:
                    break
            
            current_offset += fetch_size
            logger.info(f"Fetched {len(all_pairs)}/{limit} pairs...")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    return all_pairs


def download_image(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a single image."""
    try:
        if output_path.exists():
            return True
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return False


def get_filename_from_url(url: str, prefix: str) -> str:
    """Generate a unique filename from URL."""
    # Use hash of URL for unique, consistent naming
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
    # Get extension from URL
    parsed = urlparse(url)
    path = parsed.path
    ext = Path(path).suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        ext = '.jpg'
    
    return f"{prefix}_{url_hash}{ext}"


def download_pair(pair: dict, idx: int) -> dict:
    """Download a single LR/HR pair."""
    pair_id = pair["id"]
    original_url = pair["original_url"]
    upscaled_url = pair["upscaled_url"]
    
    # Generate filenames
    base_name = f"{idx:06d}"
    lr_filename = f"{base_name}.jpg"
    hr_filename = f"{base_name}.jpg"
    
    lr_path = LR_DIR / lr_filename
    hr_path = HR_DIR / hr_filename
    
    # Download both
    lr_ok = download_image(original_url, lr_path)
    hr_ok = download_image(upscaled_url, hr_path)
    
    if lr_ok and hr_ok:
        return {"success": True, "lr": lr_filename, "hr": hr_filename}
    else:
        # Clean up partial downloads
        if lr_path.exists():
            lr_path.unlink()
        if hr_path.exists():
            hr_path.unlink()
        return {"success": False}


def main():
    parser = argparse.ArgumentParser(description="Download training image pairs")
    parser.add_argument("--limit", type=int, default=10000, help="Number of pairs to download")
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel workers")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    args = parser.parse_args()
    
    logger.info(f"Fetching up to {args.limit} image pairs from Supabase...")
    pairs = fetch_image_pairs(limit=args.limit, offset=args.offset)
    logger.info(f"Found {len(pairs)} valid pairs")
    
    if not pairs:
        logger.error("No pairs found. Check SUPABASE_KEY and connectivity.")
        sys.exit(1)
    
    # Download in parallel
    logger.info(f"Downloading with {args.workers} workers...")
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_pair, pair, idx): idx 
            for idx, pair in enumerate(pairs)
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                successful += 1
            else:
                failed += 1
            
            total = successful + failed
            if total % 500 == 0:
                logger.info(f"Progress: {total}/{len(pairs)} ({successful} ok, {failed} failed)")
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"LR images: {LR_DIR}")
    logger.info(f"HR images: {HR_DIR}")
    
    # Save metadata
    meta_file = DATASET_DIR / "download_meta.json"
    with open(meta_file, 'w') as f:
        json.dump({
            "total_pairs": successful,
            "limit_requested": args.limit,
            "failed": failed
        }, f, indent=2)
    
    logger.info(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()


