#!/usr/bin/env python3
"""
Similarity Search API - Low-latency backend for "More Like This" feature.

Architecture (Variant Problem Solution):
1. Over-fetch: Query top 100 neighbors via raw HNSW search (~7ms)
2. Deduplicate: Client-side grouping by product_group_id (~1ms)
3. Collapse: Keep only highest-scoring item per group
4. Slice: Return top N unique results

SQL Functions:
- search_voyage_raw: Voyage embeddings (1024d), no SQL dedup
- search_dinov2_raw: DINOv2 embeddings (384d), no SQL dedup

Response format: [{id, similarity_score, thumbnail_url}]
Target latency: < 300ms

Run locally:
  source venv/bin/activate
  uvicorn api.local_latency_api:app --host 0.0.0.0 --port 8010
"""
import os
import time
import asyncpg
from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env so we can reuse the same variables as database/main.py
load_dotenv()


def build_dsn_from_env() -> str:
    """
    Direct connection DSN builder using env vars (same as Direct_connection.py):
    user, password, host, port, dbname
    Uses direct IPv4 connection (not pooler) for lower latency.
    """
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port", "5432")  # Direct connection port (default Postgres)
    dbname = os.getenv("dbname", "postgres")
    if not all([user, password, host]):
        raise RuntimeError("Set DIRECT_DSN or env vars: user, password, host (and optionally port, dbname)")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"


# Use direct connection (dedicated IPv4) instead of pooler for lower latency
DIRECT_DSN = os.getenv("DIRECT_DSN") or build_dsn_from_env()

POOL_MIN = int(os.getenv("POOL_MIN", "5"))
POOL_MAX = int(os.getenv("POOL_MAX", "10"))


class SearchResult(BaseModel):
    id: str
    name: str
    product_type: str
    product_group_id: str | None
    image_url: str | None
    similarity: float




app = FastAPI(title="Local Low-Latency Similarity API (Direct IPv4 Connection)")
db_pool: asyncpg.pool.Pool | None = None


async def init_connection(conn):
    """Initialize each connection with optimal HNSW search parameters."""
    await conn.execute("SET hnsw.ef_search = 40")


async def create_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            DIRECT_DSN,
            min_size=POOL_MIN,
            max_size=POOL_MAX,
            timeout=30,
            command_timeout=30,
            init=init_connection,  # Set hnsw.ef_search on each connection
        )
    return db_pool


async def warm_hnsw(pool: asyncpg.pool.Pool):
    """
    Warm up HNSW indexes and set optimal search parameters.
    - SET hnsw.ef_search = 40 (optimal for speed/recall tradeoff)
    - Touch both indexes to load pages into cache
    """
    sample_id = "00378cec-f182-4331-ab51-8ff13b2d99c1"
    try:
        async with pool.acquire() as conn:
            # Set optimal HNSW search parameter (lower = faster, higher = more accurate)
            await conn.execute("SET hnsw.ef_search = 40")
            # Warm up indexes
            await conn.fetch("SELECT * FROM search_voyage_raw($1, 1)", sample_id)
            await conn.fetch("SELECT * FROM search_dinov2_raw($1, 1)", sample_id)
            print("HNSW warmup complete (ef_search=40)")
    except Exception as e:
        print(f"Warm-up skipped: {e}")


@app.on_event("startup")
async def startup_event():
    pool = await create_pool()
    await warm_hnsw(pool)


@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None


async def run_search_raw(conn, rpc_name: str, query_id: str, over_fetch: int = 100):
    """
    Execute raw HNSW search without SQL-side deduplication.
    Over-fetches results for client-side deduplication.
    """
    db_start = time.perf_counter()
    rows = await conn.fetch(f"SELECT * FROM {rpc_name}($1, $2)", query_id, over_fetch)
    db_ms = (time.perf_counter() - db_start) * 1000
    results = [
        SearchResult(
            id=r["id"],
            name=r["name"],
            product_type=r["product_type"],
            product_group_id=r.get("product_group_id"),
            image_url=r.get("image_url"),
            similarity=float(r["similarity"]),
        )
        for r in rows
    ]
    return results, db_ms


def dedupe_by_product_group(results: List[SearchResult], limit: int) -> List[SearchResult]:
    """
    Client-side deduplication by product_group_id.
    Keeps only the highest-scoring item from each group.
    ~1ms execution time (vs ~150ms in SQL).
    """
    seen_groups = set()
    unique = []
    for item in results:
        group_key = item.product_group_id or item.id
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            unique.append(item)
            if len(unique) >= limit:
                break
    return unique


@app.get("/voyage")
async def voyage(
    id: str = Query(..., description="Product ID to find similar products for"),
    limit: int = Query(10, ge=1, le=20, description="Number of results (10-20)"),
    type: str = Query("material", description="Product type: 'material' or 'product'"),
):
    """
    Similarity search using Voyage embeddings (1024d).
    Uses over-fetch + client-side deduplication for the variant problem.
    Returns: [{id, similarity_score, thumbnail_url}]
    """
    start = time.perf_counter()
    pool = await create_pool()
    
    try:
        async with pool.acquire() as conn:
            # Step 1: Over-fetch 100 raw results (no SQL dedup)
            raw_results, db_ms = await run_search_raw(conn, "search_voyage_raw", id, over_fetch=100)
        
        # Step 2: Client-side deduplication (~1ms)
        dedup_start = time.perf_counter()
        unique_results = dedupe_by_product_group(raw_results, limit)
        dedup_ms = (time.perf_counter() - dedup_start) * 1000
        
        total_ms = (time.perf_counter() - start) * 1000
        
        # Step 3: Format response per spec: {id, similarity_score, thumbnail_url}
        response = [
            {
                "id": r.id,
                "similarity_score": round(r.similarity, 4),
                "thumbnail_url": r.image_url
            }
            for r in unique_results
        ]
        
        return response
        
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dinov2")
async def dinov2(
    id: str = Query(..., description="Product ID to find similar products for"),
    limit: int = Query(10, ge=1, le=20, description="Number of results (10-20)"),
    type: str = Query("material", description="Product type: 'material' or 'product'"),
):
    """
    Similarity search using DINOv2 embeddings (384d).
    Uses over-fetch + client-side deduplication for the variant problem.
    Returns: [{id, similarity_score, thumbnail_url}]
    Note: DINOv2 currently only supports materials.
    """
    start = time.perf_counter()
    pool = await create_pool()
    
    try:
        async with pool.acquire() as conn:
            # Step 1: Over-fetch 100 raw results (no SQL dedup)
            raw_results, db_ms = await run_search_raw(conn, "search_dinov2_raw", id, over_fetch=100)
        
        # Step 2: Client-side deduplication (~1ms)
        dedup_start = time.perf_counter()
        unique_results = dedupe_by_product_group(raw_results, limit)
        dedup_ms = (time.perf_counter() - dedup_start) * 1000
        
        total_ms = (time.perf_counter() - start) * 1000
        
        # Step 3: Format response per spec: {id, similarity_score, thumbnail_url}
        response = [
            {
                "id": r.id,
                "similarity_score": round(r.similarity, 4),
                "thumbnail_url": r.image_url
            }
            for r in unique_results
        ]
        
        return response
        
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    ok = False
    size = 0
    try:
        pool = await create_pool()
        size = pool.get_size()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        ok = True
    except Exception as e:
        return {"status": "degraded", "error": str(e), "pool_size": size}
    return {"status": "healthy", "pool_size": size}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# # # Test /voyage endpoint
# curl -X GET "http://localhost:8000/voyage?id=0001b8a9-c531-4151-90f9-8c07b47d4e7d&limit=10&type=material" \
#   -H "Content-Type: application/json" \
#   -w "\n\nTime: %{time_total}s\n"

# # Test /dinov2 endpoint
# curl -X GET "http://localhost:8000/dinov2?id=0001b8a9-c531-4151-90f9-8c07b47d4e7d&limit=10&type=material" \
#   -H "Content-Type: application/json" \
#   -w "\n\nTime: %{time_total}s\n"

# # Test with different product ID
# curl -X GET "http://localhost:8000/voyage?id=0007f879-96b0-42b2-859c-b8296d7d77eb&limit=15" \
#   -H "Content-Type: application/json" \
#   -w "\n\nTime: %{time_total}s\n"

# # Health check
# curl -X GET "http://localhost:8000/health"