#!/usr/bin/env python3
"""
DINOv2 Embedding Generator for Mattoboard
==========================================
Run on Lambda Labs GPU instance (A10/A100)

Usage:
    1. Copy this script to Lambda instance
    2. Set environment variables: SUPABASE_URL, SUPABASE_KEY
    3. Run: python generate_dinov2_embeddings.py

Output: dinov2_embeddings.json with all embeddings
"""

import os
import sys
import json
import time
import hashlib
import requests
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# ============= LOGGING =============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dinov2_generation.log")
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIG =============
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://glfevldtqujajsalahxd.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # anon key

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
OUTPUT_FILE = OUTPUT_DIR / "dinov2_embeddings.json"
FAILED_FILE = OUTPUT_DIR / "dinov2_failed.json"

BATCH_SIZE = 64  # GPU batch size
DOWNLOAD_WORKERS = 16  # Parallel image downloads
CHECKPOINT_INTERVAL = 500  # Save progress every N products


# ============= MODEL SETUP =============
def load_dinov2_model():
    """Load DINOv2 ViT-S/14 model."""
    logger.info("Loading DINOv2 ViT-S/14 model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Load model from torch hub
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Embedding dimension: 384")
    return model, device


def get_image_transform():
    """Get DINOv2 image preprocessing transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============= DATA FETCHING =============
def fetch_all_products() -> List[Dict]:
    """Fetch all material products from Supabase."""
    logger.info("Fetching products from Supabase...")
    
    all_products = []
    offset = 0
    limit = 1000
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    
    while True:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/products_search",
            params={
                "select": "id,name,materialData,mesh,productType,supplier,description",
                "productType": "eq.material",
                "objectStatus": "in.(APPROVED,APPROVED_PRO)",
                "offset": offset,
                "limit": limit
            },
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch: {response.text}")
            break
        
        batch = response.json()
        if not batch:
            break
        
        all_products.extend(batch)
        offset += limit
        logger.info(f"Fetched {len(all_products)} products...")
    
    logger.info(f"Total products fetched: {len(all_products)}")
    return all_products


def get_image_url(product: Dict) -> Optional[str]:
    """Extract image URL from product data."""
    # Try materialData first (for materials)
    if product.get("materialData"):
        files = product["materialData"].get("files", {})
        if files.get("color_original"):
            url = files["color_original"]
            # Handle relative URLs
            if url.startswith("materials/") or url.startswith("products/"):
                return f"https://storage.googleapis.com/mattoboard-assets/{url}"
            return url
    
    # Try mesh (for 3D products)
    if product.get("mesh"):
        if product["mesh"].get("rendered_image"):
            return product["mesh"]["rendered_image"]
    
    return None


# ============= IMAGE PROCESSING =============
def download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    """Download and open an image from URL."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
    except Exception as e:
        pass
    return None


def download_batch(items: List[Tuple[str, str]]) -> List[Tuple[str, Image.Image]]:
    """Download multiple images in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_image, url): (pid, url) for pid, url in items}
        for future in futures:
            pid, url = futures[future]
            try:
                img = future.result()
                if img is not None:
                    results.append((pid, img))
            except Exception:
                pass
    
    return results


@torch.no_grad()
def generate_embeddings_batch(
    model: torch.nn.Module,
    images: List[Image.Image],
    transform,
    device: torch.device
) -> torch.Tensor:
    """Generate embeddings for a batch of images."""
    # Preprocess images
    tensors = [transform(img) for img in images]
    batch = torch.stack(tensors).to(device)
    
    # Generate embeddings
    embeddings = model(batch)
    
    return embeddings.cpu()


# ============= CHECKPOINT MANAGEMENT =============
def load_checkpoint() -> Dict:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: {len(checkpoint.get('completed', []))} products done")
            return checkpoint
    return {"completed": [], "embeddings": {}, "failed": []}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def save_final_output(embeddings: Dict, failed: List[str], start_time: float):
    """Save final embeddings and failed list."""
    elapsed = time.time() - start_time
    
    # Save embeddings in format compatible with upload script
    output_data = {
        "model": "dinov2_vits14",
        "dimension": 384,
        "total_count": len(embeddings),
        "failed_count": len(failed),
        "generation_time_seconds": elapsed,
        "generated_at": datetime.utcnow().isoformat(),
        "embeddings": [
            {"id": pid, "embedding": emb}
            for pid, emb in embeddings.items()
        ]
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f)
    
    # Save failed list
    with open(FAILED_FILE, "w") as f:
        json.dump({"failed_ids": failed, "count": len(failed)}, f, indent=2)
    
    logger.info(f"Saved {len(embeddings)} embeddings to {OUTPUT_FILE}")
    logger.info(f"Saved {len(failed)} failed IDs to {FAILED_FILE}")


# ============= MAIN =============
def main():
    start_time = time.time()
    
    # Load model
    model, device = load_dinov2_model()
    transform = get_image_transform()
    
    # Fetch products
    products = fetch_all_products()
    
    # Prepare (id, url) pairs
    items = []
    for p in products:
        url = get_image_url(p)
        if url:
            items.append((p["id"], url))
    
    logger.info(f"Products with valid URLs: {len(items)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint.get("completed", []))
    embeddings = checkpoint.get("embeddings", {})
    failed = checkpoint.get("failed", [])
    
    # Filter out already processed
    items_to_process = [(pid, url) for pid, url in items if pid not in completed_set]
    logger.info(f"Already processed: {len(completed_set)}")
    logger.info(f"Remaining to process: {len(items_to_process)}")
    
    if not items_to_process:
        logger.info("All products already processed!")
        save_final_output(embeddings, failed, start_time)
        return
    
    # Process in batches
    processed_since_checkpoint = 0
    
    for i in tqdm(range(0, len(items_to_process), BATCH_SIZE), desc="Generating embeddings"):
        batch_items = items_to_process[i:i + BATCH_SIZE]
        
        # Download images
        downloaded = download_batch(batch_items)
        
        if not downloaded:
            # All downloads failed
            for pid, _ in batch_items:
                if pid not in failed:
                    failed.append(pid)
            continue
        
        # Extract successful downloads
        pids = [d[0] for d in downloaded]
        images = [d[1] for d in downloaded]
        
        # Track failed downloads
        downloaded_set = set(pids)
        for pid, _ in batch_items:
            if pid not in downloaded_set and pid not in failed:
                failed.append(pid)
        
        # Generate embeddings
        try:
            batch_embeddings = generate_embeddings_batch(model, images, transform, device)
            
            for pid, emb in zip(pids, batch_embeddings):
                embeddings[pid] = emb.numpy().tolist()
                completed_set.add(pid)
            
            processed_since_checkpoint += len(pids)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for pid in pids:
                if pid not in failed:
                    failed.append(pid)
        
        # Checkpoint
        if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
            checkpoint = {
                "completed": list(completed_set),
                "embeddings": embeddings,
                "failed": failed
            }
            save_checkpoint(checkpoint)
            processed_since_checkpoint = 0
            logger.info(f"Checkpoint saved: {len(embeddings)} embeddings")
    
    # Final save
    save_final_output(embeddings, failed, start_time)
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful: {len(embeddings)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Speed: {len(embeddings) / elapsed:.1f} images/sec")


if __name__ == "__main__":
    main()



