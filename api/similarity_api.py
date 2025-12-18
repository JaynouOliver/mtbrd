#!/usr/bin/env python3
"""
Similarity Search API with Mumbai Read Replica
Optimized for <200ms TTFB using asyncpg and pre-warmed connection pool.

DEPLOYMENT NOTE: For optimal <200ms latency, deploy this API in ap-south-1 (Mumbai)
to colocate with the Supabase read replica. Current network latency from other
regions adds 100-250ms overhead.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncpg
import asyncio
import time
import os

# Mumbai Read Replica Connection Config
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres.glfevldtqujajsalahxd-rr-ap-south-1-mxnjq"),
    "password": os.getenv("DB_PASSWORD", "t1PAdg7zueX6pcvb"),
    "host": os.getenv("DB_HOST", "aws-1-ap-south-1.pooler.supabase.com"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "postgres"),
}

# Global connection pool
pool: asyncpg.Pool = None


async def warmup_pool():
    """Pre-execute queries to warm up the database cache."""
    async with pool.acquire() as conn:
        # Warm up HNSW index by executing sample queries
        sample_ids = [
            '3ddac862-5cab-408e-8f48-798133bbc579',
            '40edcf6c-d4a1-4e49-92d0-7c61d68b3e3d',
        ]
        for sid in sample_ids:
            try:
                await conn.execute("SET LOCAL hnsw.ef_search = 40")
                await conn.fetch("SELECT * FROM search_similar_v4($1, 5)", sid)
                await conn.fetch("SELECT * FROM search_similar_dinov2($1, 5)", sid)
            except Exception:
                pass  # Ignore warmup errors
    print("Index cache warmed up")


async def keepalive_task():
    """Background task to keep connections warm."""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - pre-warm connection pool at startup."""
    global pool
    print("Initializing asyncpg connection pool...")
    
    # Create pool with pre-warmed connections
    pool = await asyncpg.create_pool(
        **DB_CONFIG,
        min_size=5,           # Pre-warm 5 connections
        max_size=20,          # Max 20 connections
        command_timeout=30,
        statement_cache_size=100,  # Cache prepared statements
    )
    
    # Warm up connections
    print(f"Pool initialized with {pool.get_size()} connections")
    
    # Warm up the HNSW index cache
    await warmup_pool()
    
    # Start keepalive task
    keepalive = asyncio.create_task(keepalive_task())
    
    yield
    
    # Cleanup on shutdown
    keepalive.cancel()
    await pool.close()
    print("Connection pool closed")


app = FastAPI(
    title="Material Similarity Search API",
    description="Vector similarity search using Voyage and DINOv2 embeddings via Mumbai read replica",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class SimilarProduct(BaseModel):
    id: str
    name: Optional[str]
    product_type: Optional[str]
    product_group_id: Optional[str]
    image_url: Optional[str]
    similarity: float


class SearchResponse(BaseModel):
    query_id: str
    model: str
    results: List[SimilarProduct]
    count: int
    benchmark: dict


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    replica_region: str
    pool_size: int
    connection_time_ms: float


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with connection test."""
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        connection_time = (time.perf_counter() - start) * 1000
        return HealthResponse(
            status="healthy",
            db_connected=True,
            replica_region="ap-south-1 (Mumbai)",
            pool_size=pool.get_size(),
            connection_time_ms=round(connection_time, 2)
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            db_connected=False,
            replica_region="ap-south-1 (Mumbai)",
            pool_size=0,
            connection_time_ms=0
        )


@app.get("/search/voyage", response_model=SearchResponse)
async def search_voyage(
    query_id: str = Query(..., description="Product ID to find similar products for"),
    limit: int = Query(24, ge=1, le=100, description="Number of results to return")
):
    """
    Search similar products using Voyage Multimodal-3.5 embeddings (1024d).
    Best for semantic similarity and multimodal understanding.
    """
    total_start = time.perf_counter()
    
    try:
        async with pool.acquire() as conn:
            # Set HNSW ef_search for faster queries (lower = faster, less accurate)
            await conn.execute("SET LOCAL hnsw.ef_search = 40")
            
            query_start = time.perf_counter()
            rows = await conn.fetch(
                "SELECT * FROM search_similar_v4($1, $2)",
                query_id, limit
            )
            query_time = (time.perf_counter() - query_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        results = [
            SimilarProduct(
                id=row[0],
                name=row[1],
                product_type=row[2],
                product_group_id=row[3],
                image_url=row[4],
                similarity=round(row[5], 4) if row[5] else 0
            )
            for row in rows
        ]
        
        return SearchResponse(
            query_id=query_id,
            model="voyage-multimodal-3.5",
            results=results,
            count=len(results),
            benchmark={
                "db_query_ms": round(query_time, 2),
                "total_handler_ms": round(total_time, 2),
                "embedding_dimensions": 1024,
                "replica_region": "ap-south-1",
                "pool_size": pool.get_size(),
                "note": "Deploy API in ap-south-1 for <100ms latency"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/dinov2", response_model=SearchResponse)
async def search_dinov2(
    query_id: str = Query(..., description="Product ID to find similar products for"),
    limit: int = Query(24, ge=1, le=100, description="Number of results to return")
):
    """
    Search similar products using DINOv2 ViT-S/14 embeddings (384d).
    Best for texture and surface detail similarity.
    """
    total_start = time.perf_counter()
    
    try:
        async with pool.acquire() as conn:
            # Set HNSW ef_search for faster queries
            await conn.execute("SET LOCAL hnsw.ef_search = 40")
            
            query_start = time.perf_counter()
            rows = await conn.fetch(
                "SELECT * FROM search_similar_dinov2($1, $2)",
                query_id, limit
            )
            query_time = (time.perf_counter() - query_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        results = [
            SimilarProduct(
                id=row[0],
                name=row[1],
                product_type=row[2],
                product_group_id=row[3],
                image_url=row[4],
                similarity=round(row[5], 4) if row[5] else 0
            )
            for row in rows
        ]
        
        return SearchResponse(
            query_id=query_id,
            model="dinov2-vit-s14",
            results=results,
            count=len(results),
            benchmark={
                "db_query_ms": round(query_time, 2),
                "total_handler_ms": round(total_time, 2),
                "embedding_dimensions": 384,
                "replica_region": "ap-south-1",
                "pool_size": pool.get_size(),
                "note": "Deploy API in ap-south-1 for <100ms latency"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/compare", response_model=dict)
async def compare_models(
    query_id: str = Query(..., description="Product ID to find similar products for"),
    limit: int = Query(24, ge=1, le=100, description="Number of results to return")
):
    """
    Compare both Voyage and DINOv2 results side-by-side with benchmarks.
    """
    total_start = time.perf_counter()
    
    try:
        async with pool.acquire() as conn:
            # Set HNSW ef_search for faster queries
            await conn.execute("SET LOCAL hnsw.ef_search = 40")
            
            # Voyage search
            voyage_start = time.perf_counter()
            voyage_rows = await conn.fetch(
                "SELECT * FROM search_similar_v4($1, $2)",
                query_id, limit
            )
            voyage_time = (time.perf_counter() - voyage_start) * 1000
            
            # DINOv2 search
            dinov2_start = time.perf_counter()
            dinov2_rows = await conn.fetch(
                "SELECT * FROM search_similar_dinov2($1, $2)",
                query_id, limit
            )
            dinov2_time = (time.perf_counter() - dinov2_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        def format_results(rows):
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "product_type": row[2],
                    "product_group_id": row[3],
                    "image_url": row[4],
                    "similarity": round(row[5], 4) if row[5] else 0
                }
                for row in rows
            ]
        
        return {
            "query_id": query_id,
            "voyage": {
                "model": "voyage-multimodal-3.5",
                "dimensions": 1024,
                "results": format_results(voyage_rows),
                "count": len(voyage_rows),
                "db_query_ms": round(voyage_time, 2)
            },
            "dinov2": {
                "model": "dinov2-vit-s14",
                "dimensions": 384,
                "results": format_results(dinov2_rows),
                "count": len(dinov2_rows),
                "db_query_ms": round(dinov2_time, 2)
            },
            "benchmark": {
                "voyage_query_ms": round(voyage_time, 2),
                "dinov2_query_ms": round(dinov2_time, 2),
                "total_handler_ms": round(total_time, 2),
                "replica_region": "ap-south-1",
                "pool_size": pool.get_size(),
                "note": "Deploy API in ap-south-1 for <100ms total latency"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
