#!/bin/bash
# Lambda Labs DINOv2 Setup Script
# ================================
# Run this on your Lambda instance to generate DINOv2 embeddings

set -e

echo "=============================================="
echo "DINOV2 EMBEDDING GENERATION SETUP"
echo "=============================================="

# Create working directory
mkdir -p ~/dinov2_embeddings
cd ~/dinov2_embeddings

echo ""
echo "Step 1: Setting up environment..."
echo "=============================================="

# Install required packages (Lambda instances have PyTorch pre-installed)
pip install --quiet tqdm pillow requests

# Verify GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Step 2: Creating Python script..."
echo "=============================================="

# Create the Python script
cat > generate_dinov2.py << 'PYTHONSCRIPT'
#!/usr/bin/env python3
"""
DINOv2 Embedding Generator - Lambda Labs Version
"""

import os
import json
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("generation.log")]
)
logger = logging.getLogger(__name__)

# CONFIG
SUPABASE_URL = "https://glfevldtqujajsalahxd.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY environment variable must be set")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
OUTPUT_FILE = OUTPUT_DIR / "dinov2_embeddings.json"

BATCH_SIZE = 64
DOWNLOAD_WORKERS = 16
CHECKPOINT_INTERVAL = 1000


def load_model():
    logger.info("Loading DINOv2 ViT-S/14...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    return model, device


def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def fetch_products() -> List[Dict]:
    logger.info("Fetching products from Supabase...")
    products = []
    offset = 0
    
    while True:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/products_search",
            params={
                "select": "id,name,materialData,mesh",
                "productType": "eq.material",
                "objectStatus": "in.(APPROVED,APPROVED_PRO)",
                "offset": offset,
                "limit": 1000
            },
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=30
        )
        
        if resp.status_code != 200:
            break
        
        batch = resp.json()
        if not batch:
            break
        
        products.extend(batch)
        offset += 1000
        logger.info(f"Fetched {len(products)}...")
    
    return products


def get_url(p: Dict) -> Optional[str]:
    if p.get("materialData"):
        files = p["materialData"].get("files", {})
        url = files.get("color_original")
        if url:
            if url.startswith("materials/") or url.startswith("products/"):
                return f"https://storage.googleapis.com/mattoboard-assets/{url}"
            return url
    if p.get("mesh") and p["mesh"].get("rendered_image"):
        return p["mesh"]["rendered_image"]
    return None


def download_image(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert('RGB')
    except:
        pass
    return None


def download_batch(items):
    results = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(download_image, url): (pid, url) for pid, url in items}
        for f in futures:
            pid, url = futures[f]
            try:
                img = f.result()
                if img:
                    results.append((pid, img))
            except:
                pass
    return results


@torch.no_grad()
def embed_batch(model, images, transform, device):
    tensors = [transform(img) for img in images]
    batch = torch.stack(tensors).to(device)
    return model(batch).cpu()


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "embeddings": {}, "failed": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f)


def main():
    start = time.time()
    
    model, device = load_model()
    transform = get_transform()
    
    products = fetch_products()
    items = [(p["id"], get_url(p)) for p in products if get_url(p)]
    logger.info(f"Products with URLs: {len(items)}")
    
    ckpt = load_checkpoint()
    done = set(ckpt.get("completed", []))
    embeddings = ckpt.get("embeddings", {})
    failed = ckpt.get("failed", [])
    
    todo = [(pid, url) for pid, url in items if pid not in done]
    logger.info(f"Already done: {len(done)}, Remaining: {len(todo)}")
    
    if not todo:
        logger.info("All done!")
        return
    
    count = 0
    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Generating"):
        batch_items = todo[i:i + BATCH_SIZE]
        downloaded = download_batch(batch_items)
        
        if not downloaded:
            for pid, _ in batch_items:
                if pid not in failed:
                    failed.append(pid)
            continue
        
        pids = [d[0] for d in downloaded]
        imgs = [d[1] for d in downloaded]
        
        dl_set = set(pids)
        for pid, _ in batch_items:
            if pid not in dl_set and pid not in failed:
                failed.append(pid)
        
        try:
            embs = embed_batch(model, imgs, transform, device)
            for pid, emb in zip(pids, embs):
                embeddings[pid] = emb.numpy().tolist()
                done.add(pid)
            count += len(pids)
        except Exception as e:
            logger.error(f"Error: {e}")
            for pid in pids:
                if pid not in failed:
                    failed.append(pid)
        
        if count >= CHECKPOINT_INTERVAL:
            save_checkpoint({"completed": list(done), "embeddings": embeddings, "failed": failed})
            count = 0
            logger.info(f"Checkpoint: {len(embeddings)} embeddings")
    
    # Final save
    elapsed = time.time() - start
    output = {
        "model": "dinov2_vits14",
        "dimension": 384,
        "total_count": len(embeddings),
        "failed_count": len(failed),
        "time_seconds": elapsed,
        "generated_at": datetime.utcnow().isoformat(),
        "embeddings": [{"id": pid, "embedding": emb} for pid, emb in embeddings.items()]
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(embeddings)} embeddings in {elapsed/60:.1f} min")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
PYTHONSCRIPT

echo ""
echo "Step 3: Running embedding generation..."
echo "=============================================="
echo "This will process ~51k images. Estimated time: ~30-60 minutes"
echo ""

python generate_dinov2.py

echo ""
echo "=============================================="
echo "COMPLETE!"
echo "=============================================="
echo ""
echo "Output file: ~/dinov2_embeddings/output/dinov2_embeddings.json"
echo ""
echo "To download to your local machine:"
echo "  scp ubuntu@<lambda-ip>:~/dinov2_embeddings/output/dinov2_embeddings.json ./data/"
echo ""
echo "Then upload to Supabase:"
echo "  python scripts/upload_dinov2_embeddings.py data/dinov2_embeddings.json"



