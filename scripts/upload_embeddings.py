#!/usr/bin/env python3
"""
Upload embeddings to Supabase products_search table.
Uses small batches and retry logic to handle timeouts.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 25  # Smaller batches to avoid timeout
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def upload_batch_with_retry(supabase, records, batch_num, total_batches):
    """Upload a batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            result = supabase.table("products_search").upsert(
                records,
                on_conflict="id"
            ).execute()
            return True, None
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "57014" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Batch {batch_num} timeout, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
            return False, error_msg
    return False, "Max retries exceeded"


def upload_embeddings(start_from=0):
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Load embeddings from both files
    all_embeddings = []
    
    # File 1: main embeddings
    file1 = DATA_DIR / "all_embeddings.json"
    logger.info(f"Loading {file1}...")
    with open(file1) as f:
        data1 = json.load(f)
    embeddings1 = data1.get("embeddings", [])
    logger.info(f"  Loaded {len(embeddings1)} embeddings")
    all_embeddings.extend(embeddings1)
    
    # File 2: fixed embeddings
    file2 = DATA_DIR / "fixed_embeddings.json"
    logger.info(f"Loading {file2}...")
    with open(file2) as f:
        data2 = json.load(f)
    embeddings2 = data2.get("embeddings", [])
    logger.info(f"  Loaded {len(embeddings2)} embeddings")
    all_embeddings.extend(embeddings2)
    
    logger.info(f"\nTotal embeddings: {len(all_embeddings)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    # Skip already uploaded
    if start_from > 0:
        logger.info(f"Resuming from record {start_from}")
    
    # Upload in batches
    timestamp = datetime.utcnow().isoformat()
    success_count = start_from
    error_count = 0
    failed_batches = []
    
    total_batches = (len(all_embeddings) - start_from + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(start_from, len(all_embeddings), BATCH_SIZE):
        batch = all_embeddings[i:i + BATCH_SIZE]
        batch_num = (i - start_from) // BATCH_SIZE + 1
        
        # Prepare records
        records = []
        for emb in batch:
            records.append({
                "id": emb["id"],
                "embedding_visual": emb["embedding"],
                "embedding_updated_at": timestamp
            })
        
        success, error = upload_batch_with_retry(supabase, records, batch_num, total_batches)
        
        if success:
            success_count += len(batch)
            if batch_num % 100 == 0 or batch_num == total_batches:
                logger.info(f"Progress: {batch_num}/{total_batches} batches | {success_count}/{len(all_embeddings)} records")
        else:
            error_count += len(batch)
            failed_batches.append({"start": i, "end": i + len(batch), "error": error})
            logger.error(f"Batch {batch_num} failed permanently: {error[:100]}")
        
        # Small delay between batches
        time.sleep(0.1)
    
    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if failed_batches:
        logger.info(f"\nFailed batches saved to data/failed_upload_batches.json")
        with open(DATA_DIR / "failed_upload_batches.json", "w") as f:
            json.dump(failed_batches, f, indent=2)
    
    return success_count, error_count


if __name__ == "__main__":
    # Allow resuming from a specific record
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    upload_embeddings(start_from=start)
