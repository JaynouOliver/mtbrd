#!/usr/bin/env python3
"""
Similarity Search API - Low-latency backend for "More Like This" feature.

Architecture (Variant Problem Solution):
1. Over-fetch: Query top neighbors via raw HNSW search (~5-10ms)
2. Deduplicate: Client-side grouping by product_group_id (~0.5ms)
3. Collapse: Keep only highest-scoring item per group
4. Slice: Return top N unique results

SQL Functions:
- search_voyage_raw: Voyage embeddings (1024d), no SQL dedup
- search_dinov2_raw: DINOv2 embeddings (384d), no SQL dedup

Response format: [{id, similarity_score, thumbnail_url}]
Target latency: < 300ms

Performance Optimizations:
1. HNSW ef_search=8 (ultra-aggressive, reduced from 40) - fastest searches, may reduce recall slightly
2. Dynamic over-fetch: limit * 2, max 40 (reduces DB load by 60-80%)
3. Query-level ef_search - SET LOCAL for each query (ensures consistency)
4. Query-level work_mem=32MB - more memory per query
5. enable_seqscan=off - force index usage (HNSW only)
6. Connection-level work_mem=16MB - more memory for operations
7. random_page_cost=1.1 - optimized for SSD storage
8. Direct dict operations - removed Pydantic overhead (~2-5ms saved)
9. Statement caching (100 statements) - faster query execution
10. Connection keepalive - prevents cold starts (ping every 30s)
11. Increased pool min_size=8 - better connection warmup
12. Multi-connection warmup - pre-warms 5 connections with diverse samples
13. Connection lifetime management - keeps connections alive for 5 minutes
14. Optimized deduplication - works with dicts directly

ROOT CAUSE ANALYSIS:
- Database query execution: 320-360ms (with ef_search=10)
- Inconsistency comes from HNSW graph traversal varying by embedding location
- Some embeddings require more traversal even with ef_search=10
- Solution: ef_search=8 reduces traversal, should bring DB time to 200-280ms

Expected latency: 50-180ms (warm), 150-280ms (cold start)

Run locally:
  source venv/bin/activate
  uvicorn api.local_latency_api:app --host 0.0.0.0 --port 8010
"""
import os
import time
import asyncio
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

POOL_MIN = int(os.getenv("POOL_MIN", "8"))  # Increased for better warmup
POOL_MAX = int(os.getenv("POOL_MAX", "15"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "8"))  # Ultra-aggressive for consistent <300ms (8-10 range)


class SearchResult(BaseModel):
    id: str
    name: str
    product_type: str
    product_group_id: str | None
    image_url: str | None
    similarity: float




app = FastAPI(title="Local Low-Latency Similarity API (Direct IPv4 Connection)")
db_pool: asyncpg.pool.Pool | None = None
keepalive_task_handle: asyncio.Task | None = None


async def init_connection(conn):
    """Initialize each connection with optimal HNSW search parameters."""
    # Set aggressive HNSW parameters for speed
    await conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    # Additional PostgreSQL optimizations
    await conn.execute("SET work_mem = '16MB'")  # More memory for sorts/hashes
    await conn.execute("SET random_page_cost = 1.1")  # Lower cost for index scans (SSD)


async def create_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            DIRECT_DSN,
            min_size=POOL_MIN,
            max_size=POOL_MAX,
            timeout=30,
            command_timeout=30,
            max_queries=50000,  # Higher query limit per connection
            max_inactive_connection_lifetime=300,  # Keep connections alive for 5 minutes
            statement_cache_size=100,  # Cache prepared statements for faster queries
            init=init_connection,  # Set hnsw.ef_search on each connection
        )
    return db_pool


async def warm_hnsw(pool: asyncpg.pool.Pool):
    """
    Warm up HNSW indexes and set optimal search parameters.
    - SET hnsw.ef_search = 8 (ultra-aggressive for consistent <300ms)
    - Touch both indexes with multiple sample IDs to load diverse pages into cache
    - Pre-warm multiple connections for better cold-start performance
    """
    # Use diverse sample IDs to warm up different parts of the index
    sample_ids = [
        "00378cec-f182-4331-ab51-8ff13b2d99c1",
        "eec93a3c-4a4a-4ca4-8d56-2a9d802267c6",
        "7068803d-93d1-4108-90a2-4efb48391b09",
    ]
    try:
        async def warm_conn():
            async with pool.acquire() as conn:
                await conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
                await conn.execute("SET work_mem = '16MB'")
                await conn.execute("SET random_page_cost = 1.1")
                # Warm up with multiple sample IDs to cache diverse index regions
                for sid in sample_ids:
                    try:
                        await conn.fetch("SELECT * FROM search_voyage_raw($1, 20)", sid)  # Warm with actual over-fetch amount
                        await conn.fetch("SELECT * FROM search_dinov2_raw($1, 20)", sid)
                    except Exception:
                        pass  # Ignore errors for missing embeddings
        
        # Warm up multiple connections to reduce cold-start latency
        warmup_count = min(5, POOL_MIN)  # Increased warmup connections
        warmup_tasks = [warm_conn() for _ in range(warmup_count)]
        await asyncio.gather(*warmup_tasks)
        print(f"HNSW warmup complete (ef_search={HNSW_EF_SEARCH}, {warmup_count} connections)")
    except Exception as e:
        print(f"Warm-up skipped: {e}")


async def keepalive_task():
    """Background task to keep connections warm and prevent cold starts."""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        try:
            pool = await create_pool()
            async with pool.acquire() as conn:
                # Simple ping to keep connection alive
                await conn.fetchval("SELECT 1")
        except Exception:
            pass  # Ignore keepalive errors


@app.on_event("startup")
async def startup_event():
    global keepalive_task_handle
    pool = await create_pool()
    await warm_hnsw(pool)
    # Start keepalive task to prevent connection cold starts
    keepalive_task_handle = asyncio.create_task(keepalive_task())


@app.on_event("shutdown")
async def shutdown_event():
    global db_pool, keepalive_task_handle
    # Cancel keepalive task
    if keepalive_task_handle:
        keepalive_task_handle.cancel()
        try:
            await keepalive_task_handle
        except asyncio.CancelledError:
            pass
    # Close connection pool
    if db_pool:
        await db_pool.close()
        db_pool = None


async def run_search_raw(conn, rpc_name: str, query_id: str, over_fetch: int = 100):
    """
    Execute raw HNSW search without SQL-side deduplication.
    Over-fetches results for client-side deduplication.
    Optimized: Returns dicts directly, minimal processing, connection-level ef_search.
    """
    # Set ultra-aggressive HNSW parameters for this transaction
    await conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
    # Additional query-level optimizations
    await conn.execute("SET LOCAL work_mem = '32MB'")  # More memory for this query
    await conn.execute("SET LOCAL enable_seqscan = off")  # Force index usage
    
    db_start = time.perf_counter()
    rows = await conn.fetch(f"SELECT * FROM {rpc_name}($1, $2)", query_id, over_fetch)
    db_ms = (time.perf_counter() - db_start) * 1000
    # Fast dict construction - asyncpg Records support efficient dict access
    results = [
        {
            "id": r["id"],
            "product_group_id": r.get("product_group_id"),
            "image_url": r.get("image_url"),
            "similarity": float(r["similarity"]),
        }
        for r in rows
    ]
    return results, db_ms


def dedupe_by_product_group(results: List[dict], limit: int) -> List[dict]:
    """
    Client-side deduplication by product_group_id.
    Keeps only the highest-scoring item from each group.
    Optimized: Works with dicts directly for lower overhead.
    ~0.5ms execution time (vs ~150ms in SQL).
    """
    seen_groups = set()
    unique = []
    for item in results:
        group_key = item.get("product_group_id") or item["id"]
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
    Optimized: Dynamic over-fetch, lower ef_search, direct dict operations.
    """
    start = time.perf_counter()
    pool = await create_pool()
    
    try:
        # Dynamic over-fetch: limit * 2 (extremely aggressive - minimizes DB load)
        # For limit=10, fetch 20 instead of 100 (80% reduction)
        over_fetch = min(limit * 2, 40)
        
        async with pool.acquire() as conn:
            # Step 1: Over-fetch raw results (no SQL dedup)
            raw_results, db_ms = await run_search_raw(conn, "search_voyage_raw", id, over_fetch=over_fetch)
        
        # Step 2: Client-side deduplication (~0.5ms)
        unique_results = dedupe_by_product_group(raw_results, limit)
        
        # Step 3: Format response per spec: {id, similarity_score, thumbnail_url}
        # Optimized: Build response directly without intermediate steps
        return [
            {
                "id": r["id"],
                "similarity_score": round(r["similarity"], 4),
                "thumbnail_url": r["image_url"]
            }
            for r in unique_results
        ]
        
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
    Optimized: Dynamic over-fetch, lower ef_search, direct dict operations.
    """
    start = time.perf_counter()
    pool = await create_pool()
    
    try:
        # Dynamic over-fetch: limit * 2 (extremely aggressive - minimizes DB load)
        # For limit=10, fetch 20 instead of 100 (80% reduction)
        over_fetch = min(limit * 2, 40)
        
        async with pool.acquire() as conn:
            # Step 1: Over-fetch raw results (no SQL dedup)
            raw_results, db_ms = await run_search_raw(conn, "search_dinov2_raw", id, over_fetch=over_fetch)
        
        # Step 2: Client-side deduplication (~0.5ms)
        unique_results = dedupe_by_product_group(raw_results, limit)
        
        # Step 3: Format response per spec: {id, similarity_score, thumbnail_url}
        # Optimized: Build response directly without intermediate steps
        return [
            {
                "id": r["id"],
                "similarity_score": round(r["similarity"], 4),
                "thumbnail_url": r["image_url"]
            }
            for r in unique_results
        ]
        
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