#!/usr/bin/env python3
"""
Generate embeddings for all eligible products using Voyage Multimodal 3.5.
Uses PARALLEL processing to leverage 4000 RPM rate limit.

This script:
1. Fetches all eligible products from Supabase
2. Uses image URLs directly (NO downloading needed!)
3. Runs PARALLEL API calls (up to 50 concurrent) for speed
4. Combines image + high-value metadata for rich embeddings
5. Saves all results to JSON for review before DB upload

Usage:
    python scripts/generate_all_embeddings.py                    # Full run
    python scripts/generate_all_embeddings.py --limit 100        # Test with 100 products
    python scripts/generate_all_embeddings.py --workers 50       # Parallel workers
    python scripts/generate_all_embeddings.py --resume           # Resume from checkpoint

Output: data/all_embeddings.json
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "all_embeddings.json"
CHECKPOINT_FILE = DATA_DIR / "embedding_checkpoint.json"
ERROR_LOG_FILE = DATA_DIR / "embedding_errors.log"

# Voyage settings
VOYAGE_MODEL = "voyage-multimodal-3.5"
VOYAGE_API_URL = "https://api.voyageai.com/v1/multimodalembeddings"
EMBEDDING_DIMENSION = 1024

# Rate limits (upgraded tier: 4000 RPM)
RATE_LIMIT_RPM = 4000

# Parallel settings - key for speed!
DEFAULT_WORKERS = 50  # Concurrent API calls
BATCH_SIZE = 1  # Process 1 product per API call for simpler parallelization

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1

# Thread-safe counters
lock = threading.Lock()
stats = {
    "successful": 0,
    "failed": 0,
    "processed": 0
}


# =============================================================================
# Supabase Client
# =============================================================================

def get_supabase_client():
    """Create Supabase client."""
    from supabase import create_client
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_products(supabase, limit: Optional[int] = None) -> list:
    """Fetch all eligible products."""
    all_products = []
    offset = 0
    batch_size = 500
    
    logger.info("Fetching products from Supabase...")
    
    while True:
        try:
            response = supabase.rpc(
                "export_products_for_embedding",
                {"batch_offset": offset, "batch_limit": batch_size}
            ).execute()
            
            batch = response.data
            if not batch:
                break
            
            all_products.extend(batch)
            offset += len(batch)
            
            if offset % 5000 == 0:
                logger.info(f"Fetched {len(all_products)} products...")
            
            if len(batch) < batch_size:
                break
            
            if limit and len(all_products) >= limit:
                all_products = all_products[:limit]
                break
                
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"Timeout at offset {offset}, retrying...")
                time.sleep(2)
                continue
            raise
    
    return all_products


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_high_value_metadata(product: dict) -> str:
    """Extract high-value metadata fields for embedding context."""
    parts = []
    
    try:
        name = product.get("name")
        if name:
            parts.append(str(name))
        
        material_data = product.get("material_data") or {}
        material_category = material_data.get("materialCategory")
        if material_category:
            parts.append(str(material_category))
        
        metadata = product.get("metadata_data") or {}
        
        pattern = metadata.get("pattern")
        if pattern and isinstance(pattern, list) and len(pattern) > 0:
            parts.append(str(pattern[0]))
        
        main_category = metadata.get("mainCategory")
        if main_category:
            if isinstance(main_category, list) and len(main_category) > 0:
                cat = str(main_category[0]).replace('[', '').replace(']', '').replace('"', '')
                parts.append(cat)
            elif isinstance(main_category, str):
                parts.append(main_category.replace('[', '').replace(']', '').replace('"', ''))
        
        sub_category = metadata.get("subCategory")
        if sub_category:
            parts.append(str(sub_category))
    except:
        pass
    
    return ". ".join(parts) if parts else ""


# =============================================================================
# Voyage API (Single Product)
# =============================================================================

def call_voyage_api_single(image_url: str, text_context: str = None) -> Tuple[Optional[list], Optional[str]]:
    """
    Call Voyage API for a single product.
    Returns: (embedding, error_message)
    """
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    content = []
    if text_context:
        content.append({"type": "text", "text": text_context})
    content.append({"type": "image_url", "image_url": image_url})
    
    payload = {
        "inputs": [{"content": content}],
        "model": VOYAGE_MODEL,
        "input_type": "document",
        "truncation": True
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                VOYAGE_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0]["embedding"], None
                return None, "No embedding in response"
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
                continue
            
            if response.status_code >= 500:
                # Server error - retry
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
                continue
            
            # Client error (400, etc.) - don't retry
            return None, f"HTTP {response.status_code}: {response.text[:300]}"
            
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
                continue
            return None, "Timeout"
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
                continue
            return None, str(e)
    
    return None, "Max retries exceeded"


def process_single_product(product: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Process a single product.
    Returns: (embedding_record, error_record)
    """
    product_id = product["id"]
    image_url = product.get("image_url")
    
    if not image_url:
        return None, {
            "id": product_id,
            "name": product.get("name"),
            "error": "No image URL",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    text_context = extract_high_value_metadata(product)
    
    embedding, error = call_voyage_api_single(image_url, text_context)
    
    if embedding:
        return {
            "id": product_id,
            "name": product.get("name"),
            "product_group_id": product.get("product_group_id"),
            "product_type": product.get("product_type"),
            "supplier": product.get("supplier"),
            "image_url": image_url,
            "text_context": text_context if text_context else None,
            "has_metadata": bool(text_context),
            "embedding": embedding,
            "embedding_dim": len(embedding),
        }, None
    else:
        return None, {
            "id": product_id,
            "name": product.get("name"),
            "image_url": image_url,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# Progress Management
# =============================================================================

def load_checkpoint() -> dict:
    """Load checkpoint for resuming."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"processed_ids": set()}


def save_checkpoint(processed_ids: set):
    """Save checkpoint."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"processed_ids": list(processed_ids)}, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def save_results(embeddings: list, failed: list, metadata: dict):
    """Save embeddings and errors to JSON file."""
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": VOYAGE_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_embeddings": len(embeddings),
        "total_failed": len(failed),
        "with_metadata_count": sum(1 for e in embeddings if e.get("has_metadata")),
        "image_only_count": sum(1 for e in embeddings if not e.get("has_metadata")),
        **metadata,
        "embeddings": embeddings,
        "failed_products": failed
    }
    
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f)
        logger.info(f"Saved {len(embeddings)} embeddings + {len(failed)} errors to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def log_error(error_info: dict):
    """Append error to error log file."""
    try:
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(json.dumps(error_info) + "\n")
    except:
        pass


# =============================================================================
# Parallel Processing Worker
# =============================================================================

def worker_process_product(product: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """Worker function for thread pool."""
    global stats
    
    try:
        emb, err = process_single_product(product)
        
        with lock:
            stats["processed"] += 1
            if emb:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
        
        return emb, err
        
    except Exception as e:
        with lock:
            stats["processed"] += 1
            stats["failed"] += 1
        
        return None, {
            "id": product.get("id"),
            "error": f"Worker error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_embedding_pipeline(
    limit: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = False
):
    """Run the full embedding generation pipeline with PARALLEL processing."""
    
    global stats
    stats = {"successful": 0, "failed": 0, "processed": 0}
    
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATION PIPELINE (PARALLEL)")
    logger.info("=" * 60)
    logger.info(f"Model: {VOYAGE_MODEL}")
    logger.info(f"Rate limit: {RATE_LIMIT_RPM} RPM")
    logger.info(f"Parallel workers: {workers}")
    logger.info(f"Using image URLs directly (no download needed)")
    
    # Initialize clients
    supabase = get_supabase_client()
    
    # Fetch products
    products = fetch_all_products(supabase, limit=limit)
    logger.info(f"Total products fetched: {len(products)}")
    
    # Filter to products with images
    products_with_images = [p for p in products if p.get("image_url")]
    products_without_images = [p for p in products if not p.get("image_url")]
    
    logger.info(f"Products with image URL: {len(products_with_images)}")
    logger.info(f"Products without image URL: {len(products_without_images)}")
    
    # Track products without images as "failed"
    all_failed = [{
        "id": p["id"],
        "name": p.get("name"),
        "error": "No image URL available",
        "timestamp": datetime.utcnow().isoformat()
    } for p in products_without_images]
    
    products = products_with_images
    
    # Resume handling
    all_embeddings = []
    processed_ids = set()
    
    if resume and CHECKPOINT_FILE.exists():
        checkpoint = load_checkpoint()
        processed_ids = set(checkpoint.get("processed_ids", []))
        
        # Load existing results
        if OUTPUT_FILE.exists():
            try:
                with open(OUTPUT_FILE) as f:
                    data = json.load(f)
                    all_embeddings = data.get("embeddings", [])
                    all_failed = data.get("failed_products", [])
            except:
                pass
        
        products = [p for p in products if p["id"] not in processed_ids]
        logger.info(f"Resuming: {len(processed_ids)} already processed, {len(products)} remaining")
    
    if not products:
        logger.info("No products to process!")
        save_results(all_embeddings, all_failed, {"status": "completed"})
        return
    
    # Time estimate with parallel processing
    # With 50 workers and ~1 second avg per request, we can do ~50/second
    # But rate limit is 4000/min = 66.67/second, so we're within limits
    estimated_seconds = len(products) / min(workers, 60)  # Conservative estimate
    estimated_minutes = estimated_seconds / 60
    logger.info(f"Products to process: {len(products)}")
    logger.info(f"Estimated time: {estimated_minutes:.1f} minutes with {workers} parallel workers")
    
    # Process in parallel
    start_time = time.time()
    batch_embeddings = []
    batch_errors = []
    
    logger.info(f"\nStarting parallel processing with {workers} workers...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_product = {
            executor.submit(worker_process_product, p): p 
            for p in products
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_product), 1):
            try:
                emb, err = future.result()
                
                if emb:
                    batch_embeddings.append(emb)
                    processed_ids.add(emb["id"])
                if err:
                    batch_errors.append(err)
                    log_error(err)
                    processed_ids.add(err["id"])
                
            except Exception as e:
                product = future_to_product[future]
                err = {
                    "id": product.get("id"),
                    "error": f"Future error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                batch_errors.append(err)
                log_error(err)
            
            # Progress logging every 500 products
            if i % 500 == 0 or i == len(products):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = len(products) - i
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {i}/{len(products)} | "
                    f"Success: {stats['successful']} | Failed: {stats['failed']} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min"
                )
                
                # Save checkpoint
                save_checkpoint(processed_ids)
            
            # Incremental save every 5000 products
            if i % 5000 == 0:
                all_embeddings.extend(batch_embeddings)
                all_failed.extend(batch_errors)
                batch_embeddings = []
                batch_errors = []
                
                save_results(all_embeddings, all_failed, {
                    "status": "in_progress",
                    "products_processed": i,
                    "total_products": len(products)
                })
    
    # Add final batch
    all_embeddings.extend(batch_embeddings)
    all_failed.extend(batch_errors)
    
    # Final save
    total_time = time.time() - start_time
    
    save_results(all_embeddings, all_failed, {
        "status": "completed",
        "total_products_attempted": len(products),
        "processing_time_seconds": total_time,
        "processing_time_minutes": total_time / 60,
        "success_rate": f"{100 * len(all_embeddings) / (len(all_embeddings) + len(all_failed)):.1f}%" if (len(all_embeddings) + len(all_failed)) > 0 else "N/A"
    })
    
    # Cleanup checkpoint on completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total embeddings generated: {len(all_embeddings)}")
    logger.info(f"With metadata: {sum(1 for e in all_embeddings if e.get('has_metadata'))}")
    logger.info(f"Image only: {sum(1 for e in all_embeddings if not e.get('has_metadata'))}")
    logger.info(f"Failed: {len(all_failed)}")
    logger.info(f"Success rate: {100 * len(all_embeddings) / (len(all_embeddings) + len(all_failed)):.1f}%")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Throughput: {len(products) / total_time:.1f} products/second")
    logger.info(f"Output file: {OUTPUT_FILE}")
    
    file_size = OUTPUT_FILE.stat().st_size / 1024 / 1024
    logger.info(f"File size: {file_size:.1f} MB")
    
    if all_failed:
        logger.warning(f"\n⚠️  {len(all_failed)} products failed. Check 'failed_products' in {OUTPUT_FILE}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for all products (PARALLEL)")
    parser.add_argument("--limit", type=int, help="Limit number of products (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Validate environment
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY")
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    try:
        run_embedding_pipeline(
            limit=args.limit,
            workers=args.workers,
            resume=args.resume
        )
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Progress has been saved.")
        logger.info("Run with --resume to continue from where you left off.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
