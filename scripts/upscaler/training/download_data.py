#!/usr/bin/env python3
"""
Download training data from Supabase.
Downloads original (LR) and Topaz-upscaled (HR) image pairs.

Usage:
    export SUPABASE_KEY=your_key
    python download_data.py --limit 20000
"""
import os
import sys
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('download')

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://glfevldtqujajsalahxd.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
HR_DIR = Path('/home/ubuntu/upscaler/datasets/topaz_train/hr')
LR_DIR = Path('/home/ubuntu/upscaler/datasets/topaz_train/lr')
HR_DIR.mkdir(parents=True, exist_ok=True)
LR_DIR.mkdir(parents=True, exist_ok=True)


def fetch_pairs(limit):
    """Fetch image pair URLs from Supabase."""
    if not SUPABASE_KEY:
        logger.error('SUPABASE_KEY not set!')
        sys.exit(1)
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}
    pairs, offset = [], 0
    while len(pairs) < limit:
        take = min(1000, limit - len(pairs))
        url = (
            f"{SUPABASE_URL}/rest/v1/productsV2"
            f"?select=id,metadata,materialData"
            f"&productType=in.(material,fixed%20material)"
            f"&objectStatus=in.(APPROVED,APPROVED_PRO)"
            f"&order=updatedAt.desc"
            f"&limit={take}&offset={offset}"
        )
        r = requests.get(url, headers=headers, timeout=60)
        if not r.ok:
            logger.error(f'Request failed: {r.status_code}')
            break
        data = r.json()
        if not data:
            break
        for item in data:
            meta = item.get('metadata') or {}
            mat = item.get('materialData') or {}
            files = mat.get('files') or {}
            orig = meta.get('materialImageUrl')
            upsc = files.get('color_original')
            if orig and upsc and orig != upsc:
                pairs.append({'id': item.get('id'), 'orig': orig, 'upsc': upsc})
                if len(pairs) >= limit:
                    break
        offset += take
        logger.info(f'Fetched {len(pairs)}/{limit} valid pairs...')
    return pairs


def download(url, path, timeout=60):
    """Download a single file."""
    try:
        if path.exists() and path.stat().st_size > 1000:
            return True
        resp = requests.get(url, timeout=timeout, stream=True)
        if not resp.ok:
            return False
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def download_pair(idx, pair):
    """Download both LR and HR images for a pair."""
    lr_path = LR_DIR / f'{idx:06d}.jpg'
    hr_path = HR_DIR / f'{idx:06d}.jpg'
    lr_ok = download(pair['orig'], lr_path)
    hr_ok = download(pair['upsc'], hr_path)
    if not (lr_ok and hr_ok):
        for p in [lr_path, hr_path]:
            if p.exists():
                p.unlink()
        return False
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    
    pairs = fetch_pairs(args.limit)
    logger.info(f'Total valid pairs: {len(pairs)}')
    
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_pair, i, p): i for i, p in enumerate(pairs)}
        for fut in as_completed(futures):
            if fut.result():
                ok += 1
            if ok % 500 == 0 and ok > 0:
                logger.info(f'Downloaded {ok}/{len(pairs)}')
    
    logger.info(f'Download complete: {ok}/{len(pairs)} pairs')

