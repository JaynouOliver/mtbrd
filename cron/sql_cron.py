#!/usr/bin/env python3
"""
Generate Voyage embeddings for products missing embeddings in products_search.

This script:
1. Finds products in products_search table that are missing embeddings
2. Generates Voyage embeddings using embed_image_url
3. Updates products_search table with embeddings
4. Logs all changes for visibility

Simple approach: Just check products_search table directly.
"""

import os
import sys
import logging
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.voyage_embedding import embed_image_url

DB_CONFIG = {
    "primary_db_user": os.getenv("PRIMARY_DB_USER"),
    "primary_db_password": os.getenv("PRIMARY_DB_PASSWORD"),
    "primary_db_host": os.getenv("PRIMARY_DB_HOST"),
    "primary_db_port": os.getenv("PRIMARY_DB_PORT"),
    "primary_db_name": os.getenv("PRIMARY_DB_NAME")
}

def get_db_connection():
    try:
        connection = psycopg2.connect(
            user=DB_CONFIG["primary_db_user"],
            password=DB_CONFIG["primary_db_password"],
            host=DB_CONFIG["primary_db_host"],
            port=DB_CONFIG["primary_db_port"],
            dbname=DB_CONFIG["primary_db_name"]
        )
        return connection
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise


def normalize_image_url(url):
    """
    Normalize image URL - handle both full URLs and relative paths.
    - If already a full URL (starts with http/https), use as-is
    - If relative path (starts with materials/ or products/), convert to full URL
    Note: Most URLs should already be full URLs. This handles legacy relative paths.
    """
    if not url:
        return None
    
    # Already a full URL - use as-is (this is the normal case)
    if url.startswith("http://") or url.startswith("https://"):
        return url
    
    # Legacy relative paths - convert to full URL
    # These are old products, new products should have full URLs
    if url.startswith("materials/") or url.startswith("products/"):
        # Try the mattoboard-b8284.appspot.com bucket first (matches your example)
        return f"https://storage.googleapis.com/mattoboard-b8284.appspot.com/{url}"
    
    # Unknown format - return as-is
    return url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

# Control flag: Set to False to only detect and log changes without updating database
# Set to True to actually update the database with embeddings
UPDATE_TABLE = True

# Limit number of products to process (for testing). Set to None to process all.
# This helps avoid timeouts when testing with large datasets
PROCESS_LIMIT = 1  # Process first 10 products for testing


def find_products_needing_embeddings(connection):
    """
    Find products in products_search that need embeddings.
    Simple: Just check products_search table directly for missing embeddings.
    """
    query = """
    SELECT
      ps."id",
      ps."name",
      ps."product_group_id",
      ps."productType",
      COALESCE(
        ps.metadata->'materialData'->'files'->>'color_original',
        ps."materialData"->'files'->>'color_original',
        ps."materialData"->>'renderedImage'
      ) AS upscaled_image,
      'MISSING_EMBEDDING' AS change_type
    FROM public.products_search ps
    WHERE
      ps."productType" IN ('material', 'paint')
      AND ps."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
      AND ps.embedding_visual IS NULL
      AND (
        ps.metadata->'materialData'->'files'->>'color_original' IS NOT NULL
        OR ps."materialData"->'files'->>'color_original' IS NOT NULL
        OR ps."materialData"->>'renderedImage' IS NOT NULL
      )
    ORDER BY ps."id"
    LIMIT %s
    """
    
    cursor = connection.cursor()
    try:
        # Use a reasonable limit to avoid timeouts (can be overridden)
        limit = PROCESS_LIMIT if PROCESS_LIMIT else 1000000  # Very large number if None
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        products = []
        for row in rows:
            products.append({
                "id": row[0],
                "name": row[1],
                "product_group_id": row[2],
                "productType": row[3],
                "upscaled_image": row[4],
                "change_type": row[5]  # Always 'MISSING_EMBEDDING'
            })
        
        return products
    finally:
        cursor.close()


def process_products(connection, products, update_table=False):
    """
    Process products: generate embeddings and optionally update database.
    
    Args:
        connection: Database connection
        products: List of products to process
        update_table: If True, updates database. If False, only logs changes.
    """
    total = len(products)
    success_count = 0
    error_count = 0
    errors = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING {total} PRODUCTS")
    if not update_table:
        logger.info("DRY RUN MODE: Changes will be logged but NOT saved to database")
    logger.info(f"{'='*60}\n")
    
    cursor = connection.cursor() if update_table else None
    
    for i, product in enumerate(products, 1):
        product_id = product["id"]
        product_name = product.get("name", "N/A")
        url = product["upscaled_image"]
        change_type = product["change_type"]
        
        logger.info(f"[{i}/{total}] {change_type}: {product_id} - {product_name[:50]}")
        if url:
            logger.info(f"  URL: {url[:80]}...")
        else:
            logger.info(f"  URL: None")
        
        if not url:
            logger.warning(f"  SKIP: No URL")
            error_count += 1
            errors.append({"id": product_id, "error": "No URL"})
            continue
        
        # Normalize URL (convert relative paths to full URLs if needed)
        normalized_url = normalize_image_url(url)
        if normalized_url != url:
            logger.info(f"  Normalized URL: {normalized_url[:80]}...")
        
        try:
            # Generate embedding for NEW or MISSING_EMBEDDING products
            emb = embed_image_url(normalized_url)
            logger.info(f"  Generated embedding: {len(emb)} dimensions")
            
            if update_table:
                # Convert embedding list to PostgreSQL vector format
                embedding_str = '[' + ','.join(map(str, emb)) + ']'
                
                # Upsert into products_search (only embedding_visual, no URL tracking)
                upsert_query = """
                INSERT INTO public.products_search (
                    id, 
                    embedding_visual, 
                    embedding_updated_at
                )
                VALUES (%s, %s::vector(1024), %s)
                ON CONFLICT (id) 
                DO UPDATE SET
                    embedding_visual = EXCLUDED.embedding_visual,
                    embedding_updated_at = EXCLUDED.embedding_updated_at
                """
                
                cursor.execute(
                    upsert_query,
                    (product_id, embedding_str, datetime.utcnow())
                )
                connection.commit()
                
                logger.info(f"  SUCCESS: Updated products_search with embedding")
            else:
                logger.info(f"  DRY RUN: Would update products_search (skipped)")
            
            success_count += 1
            
        except Exception as e:
            if update_table:
                connection.rollback()
            logger.error(f"  ERROR: {str(e)}")
            error_count += 1
            errors.append({
                "id": product_id,
                "name": product_name,
                "url": url,
                "error": str(e)
            })
    
    if cursor:
        cursor.close()
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total processed: {total}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if errors:
        logger.warning(f"\nFailed products:")
        for err in errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {err['id']}: {err.get('error', 'Unknown')}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more")
    
    return success_count, error_count, errors


def main():
    """Main function."""
    logger.info("="*60)
    logger.info("PRODUCT EMBEDDING CHANGE TRACKER")
    logger.info("="*60)
    
    # Validate environment
    if not VOYAGE_API_KEY:
        logger.error("Missing environment variable: VOYAGE_API_KEY")
        sys.exit(1)
    
    connection = None
    try:
        connection = get_db_connection()
        logger.info("Connected to database")
        
        # First, get total count without limit
        logger.info("\nAnalyzing changes...")
        cursor = connection.cursor()
        try:
            count_query = """
            SELECT COUNT(*)
            FROM public.products_search ps
            WHERE
              ps."productType" IN ('material', 'paint')
              AND ps."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
              AND ps.embedding_visual IS NULL
              AND (
                ps.metadata->'materialData'->'files'->>'color_original' IS NOT NULL
                OR ps."materialData"->'files'->>'color_original' IS NOT NULL
                OR ps."materialData"->>'renderedImage' IS NOT NULL
              )
            """
            cursor.execute(count_query)
            total_count = cursor.fetchone()[0]
        finally:
            cursor.close()
        
        if total_count == 0:
            logger.info("\nNo products missing embeddings. All products are up to date!")
            return
        
        logger.info(f"\nFound {total_count} products missing embeddings")
        
        # Now get products to process (with limit if set)
        products = find_products_needing_embeddings(connection)
        
        if not products:
            logger.info("No products to process!")
            return
        
        # Apply limit if set
        if PROCESS_LIMIT and len(products) > PROCESS_LIMIT:
            logger.info(f"Processing first {PROCESS_LIMIT} products (out of {total_count} total)")
            products = products[:PROCESS_LIMIT]
        else:
            logger.info(f"Processing all {len(products)} products")
        
        # Show sample
        logger.info("\nSample products:")
        for p in products[:5]:
            logger.info(f"  - {p['change_type']}: {p['id']} - {p.get('name', 'N/A')[:40]}")
        if len(products) > 5:
            logger.info(f"  ... and {len(products) - 5} more")
        
        # Process products
        success, errors, error_list = process_products(connection, products, update_table=UPDATE_TABLE)
        
        if not UPDATE_TABLE:
            logger.info("\n" + "="*60)
            logger.info("DRY RUN COMPLETE - No database changes were made")
            logger.info("Set UPDATE_TABLE = True to actually update the database")
            logger.info("="*60)
        
        logger.info("\nDone!")
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
