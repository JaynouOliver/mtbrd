#!/usr/bin/env python3
"""
Fast bulk upload using RPC function for batch updates.
"""

import json
import logging
import os
import time
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://glfevldtqujajsalahxd.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable must be set")

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 100  # Records per RPC call
CONCURRENT_BATCHES = 10  # Parallel RPC calls


async def upload_batch(session, semaphore, batch, batch_num):
    """Upload a batch using RPC."""
    async with semaphore:
        url = f"{SUPABASE_URL}/rest/v1/rpc/bulk_update_dinov2"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
        }
        
        # Format for RPC: array of {id, embedding}
        updates = [{"id": e["id"], "embedding": e["embedding"]} for e in batch]
        
        for attempt in range(3):
            try:
                async with session.post(
                    url, 
                    json={"updates": updates}, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        return (batch_num, len(batch), True, None)
                    else:
                        text = await resp.text()
                        if attempt < 2:
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                        return (batch_num, len(batch), False, f"HTTP {resp.status}: {text[:100]}")
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                return (batch_num, len(batch), False, str(e)[:100])
        
        return (batch_num, len(batch), False, "Max retries")


async def main():
    input_file = DATA_DIR / "dinov2_embeddings.json"
    checkpoint_file = DATA_DIR / "upload_rpc_checkpoint.json"
    
    logger.info(f"Loading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    embeddings = data.get("embeddings", [])
    total = len(embeddings)
    
    # Load checkpoint
    start_from = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            cp = json.load(f)
            start_from = cp.get("last_idx", 0)
            logger.info(f"Resuming from {start_from}")
    
    remaining = embeddings[start_from:]
    logger.info(f"Total: {total}, Remaining: {len(remaining)}")
    logger.info(f"Batch size: {BATCH_SIZE}, Concurrent: {CONCURRENT_BATCHES}")
    
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_BATCHES)
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create batches
        batches = []
        for i in range(0, len(remaining), BATCH_SIZE):
            batches.append(remaining[i:i + BATCH_SIZE])
        
        logger.info(f"Processing {len(batches)} batches...")
        
        # Process in waves to show progress
        wave_size = CONCURRENT_BATCHES * 5
        for wave_start in range(0, len(batches), wave_size):
            wave = batches[wave_start:wave_start + wave_size]
            
            tasks = [
                upload_batch(session, semaphore, batch, wave_start + i)
                for i, batch in enumerate(wave)
            ]
            
            results = await asyncio.gather(*tasks)
            
            for batch_num, count, success, error in results:
                if success:
                    success_count += count
                else:
                    error_count += count
                    logger.error(f"Batch {batch_num} failed: {error}")
            
            # Progress
            elapsed = time.time() - start_time
            processed = success_count + error_count
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_count = len(remaining) - processed
            eta = remaining_count / rate if rate > 0 else 0
            pct = ((start_from + processed) / total) * 100
            
            logger.info(
                f"Progress: {start_from + processed}/{total} ({pct:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min | "
                f"Success: {success_count} | Errors: {error_count}"
            )
            
            # Checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_idx": start_from + processed}, f)
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {(success_count + error_count)/elapsed:.1f}/s")
    logger.info(f"Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    asyncio.run(main())

