#!/usr/bin/env python3
"""
Optimized Vector Search API
Uses direct PostgreSQL connection to Mumbai read replica for low latency.
Supports both Voyage (1024d) and DINOv2 (384d) embeddings.
"""
from fastapi import FastAPI, Query, HTTPException
import time
from typing import List, Optional
from datetime import datetime
import json
import hashlib
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import db_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Search API",
    description="High-performance vector similarity search using Voyage and DINOv2 embeddings",
    version="2.0.0"
)

# In-memory cache with TTL
cache = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


@app.on_event("startup")
async def startup():
    """Initialize database connection pools on startup."""
    await db_pool.connect()
    logger.info("Server started")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection pools on shutdown."""
    await db_pool.disconnect()
    logger.info("Server shutdown")


def hash_embedding(emb: List[float]) -> str:
    """Create a hash of the embedding for cache key."""
    return hashlib.md5(json.dumps([round(x, 6) for x in emb]).encode()).hexdigest()


@app.post("/search/voyage")
async def search_voyage(
    query_embedding: List[float],
    limit: int = Query(10, ge=1, le=50),
    distance_threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Search by Voyage embedding (1024 dimensions).
    
    Args:
        query_embedding: 1024-dimensional Voyage embedding vector
        limit: Maximum number of results (1-50)
        distance_threshold: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        Matching products with similarity scores and timing benchmark
    """
    start = time.time()
    cache_key = f"voyage:{hash_embedding(query_embedding)}:{limit}:{distance_threshold}"
    
    # Check cache
    if cache_key in cache:
        cached_data, cached_time = cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            elapsed = (time.time() - start) * 1000
            return {**cached_data, "cached": True, "ttfb_ms": round(elapsed, 2)}
    
    try:
        async with db_pool.replica_pool.acquire() as conn:
            db_start = time.time()
            # POST endpoint doesn't have product_id/group_id, so pass NULL
            results = await conn.fetch(
                "SELECT * FROM search_voyage_optimized($1::vector, $2, $3, NULL, NULL)",
                query_embedding, limit, distance_threshold
            )
            db_elapsed = (time.time() - db_start) * 1000
        
        response = {
            "count": len(results),
            "model": "voyage-multimodal-3.5",
            "embedding_dimensions": 1024,
            "results": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "product_type": r["product_type"],
                    "product_group_id": r["product_group_id"],
                    "supplier": r["supplier"],
                    "primary_image": r["primary_image"],
                    "color_hex": r["color_hex"],
                    "similarity": round(float(r["distance"]), 4)
                }
                for r in results
            ],
            "cached": False,
            "benchmark": {
                "db_query_ms": round(db_elapsed, 2),
                "replica_region": "ap-south-1 (Mumbai)"
            }
        }
        
        cache[cache_key] = (response, time.time())
        elapsed = (time.time() - start) * 1000
        response["ttfb_ms"] = round(elapsed, 2)
        return response
        
    except Exception as e:
        logger.error(f"Voyage search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/dinov2")
async def search_dinov2(
    query_embedding: List[float],
    limit: int = Query(10, ge=1, le=50),
    distance_threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Search by DINOv2 embedding (384 dimensions).
    
    Args:
        query_embedding: 384-dimensional DINOv2 embedding vector
        limit: Maximum number of results (1-50)
        distance_threshold: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        Matching products with similarity scores and timing benchmark
    """
    start = time.time()
    cache_key = f"dinov2:{hash_embedding(query_embedding)}:{limit}:{distance_threshold}"
    
    # Check cache
    if cache_key in cache:
        cached_data, cached_time = cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            elapsed = (time.time() - start) * 1000
            return {**cached_data, "cached": True, "ttfb_ms": round(elapsed, 2)}
    
    try:
        async with db_pool.replica_pool.acquire() as conn:
            db_start = time.time()
            # POST endpoint doesn't have product_id/group_id, so pass NULL
            results = await conn.fetch(
                "SELECT * FROM search_dinov2_optimized($1::vector, $2, $3, NULL, NULL)",
                query_embedding, limit, distance_threshold
            )
            db_elapsed = (time.time() - db_start) * 1000
        
        response = {
            "count": len(results),
            "model": "dinov2-vit-s14",
            "embedding_dimensions": 384,
            "results": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "product_type": r["product_type"],
                    "product_group_id": r["product_group_id"],
                    "supplier": r["supplier"],
                    "primary_image": r["primary_image"],
                    "color_hex": r["color_hex"],
                    "similarity": round(float(r["distance"]), 4)
                }
                for r in results
            ],
            "cached": False,
            "benchmark": {
                "db_query_ms": round(db_elapsed, 2),
                "replica_region": "ap-south-1 (Mumbai)"
            }
        }
        
        cache[cache_key] = (response, time.time())
        elapsed = (time.time() - start) * 1000
        response["ttfb_ms"] = round(elapsed, 2)
        return response
        
    except Exception as e:
        logger.error(f"DINOv2 search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint with connection pool status."""
    pool_size = 0
    replica_connected = False
    
    try:
        if db_pool.replica_pool:
            pool_size = db_pool.replica_pool.get_size()
            async with db_pool.replica_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            replica_connected = True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    
    return {
        "status": "healthy" if replica_connected else "degraded",
        "replica_connected": replica_connected,
        "replica_region": "ap-south-1 (Mumbai)",
        "pool_size": pool_size,
        "cache_size": len(cache),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/search/voyage/{product_id}")
async def search_voyage_by_id(
    product_id: str,
    limit: int = Query(10, ge=1, le=50),
    distance_threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Search similar products by product ID using Voyage embeddings.
    Fetches the embedding from the database and performs similarity search.
    """
    start = time.time()
    
    try:
        async with db_pool.replica_pool.acquire() as conn:
            # First, get the embedding AND product_group_id for the product
            embedding_row = await conn.fetchrow(
                "SELECT embedding_visual, product_group_id FROM products_search WHERE id = $1",
                product_id
            )
            
            if not embedding_row or not embedding_row['embedding_visual']:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Product {product_id} not found or has no Voyage embedding"
                )
            
            # Get embedding and product_group_id
            query_embedding = embedding_row['embedding_visual']
            query_group_id = embedding_row['product_group_id']
            
            # Now perform the search - exclude same product and same product_group_id
            db_start = time.time()
            results = await conn.fetch(
                "SELECT * FROM search_voyage_optimized($1::vector, $2, $3, $4, $5)",
                str(query_embedding), limit, distance_threshold, product_id, query_group_id
            )
            db_elapsed = (time.time() - db_start) * 1000
        
        response = {
            "query_id": product_id,
            "count": len(results),
            "model": "voyage-multimodal-3.5",
            "embedding_dimensions": 1024,
            "results": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "product_type": r["product_type"],
                    "product_group_id": r["product_group_id"],
                    "supplier": r["supplier"],
                    "primary_image": r["primary_image"],
                    "color_hex": r["color_hex"],
                    "similarity": round(float(r["distance"]), 4)
                }
                for r in results
            ],
            "cached": False,
            "benchmark": {
                "db_query_ms": round(db_elapsed, 2),
                "replica_region": "ap-south-1 (Mumbai)"
            }
        }
        
        elapsed = (time.time() - start) * 1000
        response["ttfb_ms"] = round(elapsed, 2)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voyage search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/dinov2/{product_id}")
async def search_dinov2_by_id(
    product_id: str,
    limit: int = Query(10, ge=1, le=50),
    distance_threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Search similar products by product ID using DINOv2 embeddings.
    Fetches the embedding from the database and performs similarity search.
    """
    start = time.time()
    
    try:
        async with db_pool.replica_pool.acquire() as conn:
            # First, get the embedding AND product_group_id for the product
            embedding_row = await conn.fetchrow(
                "SELECT embedding_dinov2, product_group_id FROM products_search WHERE id = $1",
                product_id
            )
            
            if not embedding_row or not embedding_row['embedding_dinov2']:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Product {product_id} not found or has no DINOv2 embedding"
                )
            
            # Get embedding and product_group_id
            query_embedding = embedding_row['embedding_dinov2']
            query_group_id = embedding_row['product_group_id']
            
            # Now perform the search - exclude same product and same product_group_id
            db_start = time.time()
            results = await conn.fetch(
                "SELECT * FROM search_dinov2_optimized($1::vector, $2, $3, $4, $5)",
                str(query_embedding), limit, distance_threshold, product_id, query_group_id
            )
            db_elapsed = (time.time() - db_start) * 1000
        
        response = {
            "query_id": product_id,
            "count": len(results),
            "model": "dinov2-vit-s14",
            "embedding_dimensions": 384,
            "results": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "product_type": r["product_type"],
                    "product_group_id": r["product_group_id"],
                    "supplier": r["supplier"],
                    "primary_image": r["primary_image"],
                    "color_hex": r["color_hex"],
                    "similarity": round(float(r["distance"]), 4)
                }
                for r in results
            ],
            "cached": False,
            "benchmark": {
                "db_query_ms": round(db_elapsed, 2),
                "replica_region": "ap-south-1 (Mumbai)"
            }
        }
        
        elapsed = (time.time() - start) * 1000
        response["ttfb_ms"] = round(elapsed, 2)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DINOv2 search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API documentation endpoint."""
    return {
        "service": "Vector Search API v2.0",
        "description": "High-performance similarity search for materials",
        "endpoints": {
            "GET /search/voyage/{product_id}": "Search by product ID using Voyage (1024d)",
            "GET /search/dinov2/{product_id}": "Search by product ID using DINOv2 (384d)",
            "POST /search/voyage": "Search by Voyage embedding vector",
            "POST /search/dinov2": "Search by DINOv2 embedding vector",
            "GET /health": "Health check with pool status"
        },
        "models": {
            "voyage-multimodal-3.5": {
                "dimensions": 1024,
                "description": "Multimodal embedding with semantic understanding"
            },
            "dinov2-vit-s14": {
                "dimensions": 384,
                "description": "Self-supervised vision model for texture/grain details"
            }
        },
        "replica_region": "ap-south-1 (Mumbai)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

