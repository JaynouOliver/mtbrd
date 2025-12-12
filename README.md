# Mattoboard Vector Search Infrastructure

Technical writeup for Questions 2 & 3 of the assignment.

---

## Question 2: Supabase/Postgres Architecture

### Objective

Replace Pinecone with pgvector on Supabase for vector search while maintaining Firestore (`productsV2`) as the source of truth.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Firestore (productsV2)                                                    │
│         │                                                                   │
│         │  Daily Sync (Cloud Function / Cron)                               │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Supabase Postgres                                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │  products_search (Working Table)                            │    │   │
│   │  │  ├── All productsV2 columns (id, name, supplier, etc.)      │    │   │
│   │  │  ├── embedding_visual      (vector 1024) ◄── Voyage AI      │    │   │
│   │  │  ├── embedding_semantic    (vector 1024)     [reserved]     │    │   │
│   │  │  └── embedding_updated_at  (timestamp)                      │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   │                           │                                         │   │
│   │                           │ HNSW Index                              │   │
│   │                           ▼                                         │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │  RPC Functions                                              │    │   │
│   │  │  ├── search_similar_v4()   → Fast similarity search         │    │   │
│   │  │  └── get_material_details()→ Product lookup                 │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           │ PostgREST API                                   │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Edge Function: /products                                           │   │
│   │  ├── GET /products              → List (paginated, filterable)      │   │
│   │  ├── GET /products/:id          → Single product                    │   │
│   │  ├── GET /products/stats        → Embedding statistics              │   │
│   │  └── GET /products/similarity/:id → Similar products                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sync Strategy: Firestore → Postgres (this is not implemented , as there is no firestore)

**Approach:** Incremental daily sync using `updatedAt` timestamp.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SYNC PIPELINE (Daily Cron)                                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Query Firestore:                                                     │
│     WHERE updatedAt > last_sync_timestamp                                │
│                                                                          │
│  2. Upsert to Postgres:                                                  │
│     INSERT ... ON CONFLICT (id) DO UPDATE                                │
│                                                                          │
│  3. Flag for re-embedding:                                               │
│     SET embedding_updated_at = NULL for changed image fields             │
│                                                                          │
│  4. Trigger embedding pipeline for flagged rows                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why this approach:**
- Firestore remains source of truth (no migration risk)
- Incremental sync minimizes load (~hundreds of records/day vs 70k full scan)
- `embedding_updated_at = NULL` marks rows needing re-embedding

### Dev/Prod Environment Support

Supabase provides native branching:

| Environment | Purpose | Branch Type |
|-------------|---------|-------------|
| Production | Live data, user-facing | Main project |
| Development | Testing, schema changes | Supabase Branch |

**Workflow:**
1. Create branch: `supabase branches create --name develop`
2. Test migrations on branch
3. Merge to production: `supabase branches merge --id <branch_id>`

Branches inherit production schema but start with empty data (or can be seeded).

### Embedding Update Pipeline

**Current Implementation:** Batch pipeline with checkpointing.

```
scripts/
├── generate_all_embeddings.py   # Main embedding generator
├── upload_embeddings.py         # Push to Supabase
└── config.py                    # Configuration
```

**Pipeline Features:**
- Parallel processing (ThreadPoolExecutor, respects 4000 RPM limit)
- Checkpointing every 100 records (resume on failure)
- Error tracking with retry logic
- Handles image resolution limits (auto-resize to 4096x4096)

**Triggering Updates:**
- Manual: Run pipeline for `WHERE embedding_updated_at IS NULL`
- Automated: Cloud Function triggered by sync completion

### API Endpoints (Edge Function)

Base: `https://glfevldtqujajsalahxd.supabase.co/functions/v1/products`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/products` | GET | List products (paginated, filterable) |
| `/products/:id` | GET | Single product with computed image URL |
| `/products/filters` | GET | Available filter options |
| `/products/stats` | GET | Embedding coverage statistics |
| `/products/similarity/:id` | GET | Similar products (see Q3) |

**Authentication:** Supabase anon key in Authorization header.

---

## Question 3: Material & Product Similarity Search

### Objective

"More Like This" feature returning 10-20 visually similar products in <300ms, without returning color variants of the same product.

### Key Architectural Decision

**Split responsibilities between database and application layer:**

| Layer | Responsibility | Latency |
|-------|----------------|---------|
| Database (Postgres) | Pure vector search via HNSW | ~5ms |
| Application (Python) | Variant deduplication by product_group_id | ~1ms |

This separation achieves <300ms by keeping the database query simple and moving business logic (deduplication) to the client where it executes faster.

### 1. Embedding Model Selection

**Decision: Voyage Multimodal-3 for both materials and products.**

| Model | Strengths | Weaknesses | Decision |
|-------|-----------|------------|----------|
| **DINOv2** | Excellent texture/grain capture | Self-supervised (no semantic understanding), requires hosting | Considered(in progress) |
| **Voyage Multimodal-3** | Multimodal (image+text), managed API, 1024-dim vectors | Cost (~$0.001/image) | **Selected** |

**Rationale:**
1. **Unified model** - One embedding space for materials AND products simplifies architecture
2. **Multimodal capability** - Can incorporate metadata (name, supplier) into embedding for semantic boost
3. **Managed API** - No GPU infrastructure to maintain
4. **Proven quality** - Manual testing showed strong visual similarity for both textures and 3D objects

**Embedding Strategy:**

```python
# For products WITH rich metadata:
embedding = voyage.multimodal_embed(
    inputs=[[image, f"{name} by {supplier}. {description}"]],
    model="voyage-multimodal-3"
)

# For products with minimal metadata:
embedding = voyage.multimodal_embed(
    inputs=[[image]],
    model="voyage-multimodal-3"
)
```

**Cost:** ~$51 for 51,000 products (one-time), negligible for incremental updates.

### 2. Database & Indexing (pgvector)

**Yes, Supabase supports HNSW indexes.**

```sql
-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_embedding_visual_hnsw 
ON products_search 
USING hnsw (embedding_visual vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Partial index for filtered queries (materials only)
CREATE INDEX idx_embedding_material_approved 
ON products_search 
USING hnsw (embedding_visual vector_cosine_ops)
WHERE "productType" = 'material' 
  AND "objectStatus" IN ('APPROVED', 'APPROVED_PRO');
```

**Index Parameters:**
- `m = 16`: Connections per node (higher = better recall, more memory)
- `ef_construction = 200`: Build-time search width (higher = better index quality)

**Performance:**
- Without index: ~500ms (sequential scan)
- With HNSW index: **~5ms** (verified via `EXPLAIN ANALYZE`)

### 3. The Variant Problem (Critical Logic)

**Problem:** Nearest neighbors to "Beige Sofa" are "Black Sofa" and "Red Sofa" of the same model.

**Solution: Post-Search Diversification (Client-Side)**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DEDUPLICATION FLOW                                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. OVER-FETCH                                                           │
│     Query top 24 neighbors (2x desired limit)                            │
│     ~5ms with HNSW index                                                 │
│                                                                          │
│  2. CLIENT-SIDE DEDUPE (~1ms)                                            │
│     Group by product_group_id                                            │
│     Keep highest-scoring item per group                                  │
│                                                                          │
│  3. SLICE                                                                │
│     Return top 12 unique results                                         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why client-side instead of SQL?**

| Approach | Latency | Complexity |
|----------|---------|------------|
| SQL `DISTINCT ON` + CTE | ~150ms | High |
| Client-side Python loop | **~1ms** | Low |

**Implementation:**

```python
def dedupe_by_product_group(results: list, limit: int) -> list:
    seen_groups = set()
    unique = []
    for item in results:
        group_key = item.get('product_group_id') or item['id']
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            unique.append(item)
            if len(unique) >= limit:
                break
    return unique
```

### 4. Optimized RPC Function

```sql
CREATE OR REPLACE FUNCTION search_similar_v4(
    query_id VARCHAR,
    match_cnt INT DEFAULT 20
)
RETURNS TABLE (
    id VARCHAR,
    name VARCHAR,
    product_type VARCHAR,
    product_group_id VARCHAR,
    image_url TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    query_embedding vector(1024);
BEGIN
    -- Pre-fetch embedding (avoids correlated subquery)
    SELECT embedding_visual INTO query_embedding
    FROM products_search WHERE id = query_id;

    IF query_embedding IS NULL THEN
        RAISE EXCEPTION 'Product % has no embedding', query_id;
    END IF;

    -- Simple HNSW query with filters
    RETURN QUERY
    SELECT
        ps.id,
        ps.name,
        ps."productType",
        ps.product_group_id,
        ps."materialData"->'files'->>'color_original' AS image_url,
        1 - (ps.embedding_visual <=> query_embedding) AS similarity
    FROM products_search ps
    WHERE ps.embedding_visual IS NOT NULL
      AND ps.id != query_id
      AND ps."productType" = 'material'
      AND ps."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
    ORDER BY ps.embedding_visual <=> query_embedding
    LIMIT match_cnt;
END;
$$;
```

**Key Optimizations:**
1. Pre-fetch query embedding into variable (not subquery)
2. No SQL-side deduplication (moved to client)
3. Partial index match via `WHERE` clause filters

### 5. Two-Step Query Architecture

The similarity search is split into two distinct steps for optimal performance:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: RAW SQL QUERY (search_similar_v4)                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  What it does:                                                               │
│  - Executes pure HNSW vector search on Postgres                              │
│  - Returns top N nearest neighbors ordered by cosine similarity              │
│  - NO deduplication (may contain color variants of same product)             │
│                                                                              │
│  Why no dedup in SQL?                                                        │
│  - SQL DISTINCT ON + CTEs added ~150ms latency                               │
│  - HNSW index cannot be used efficiently with complex grouping               │
│  - Keeping the query simple = ~5ms execution time                            │
│                                                                              │
│  Output: Raw list with potential duplicates by product_group_id              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: CLIENT-SIDE DEDUPLICATION (Application Layer)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  What it does:                                                               │
│  - Receives over-fetched results (2x desired limit)                          │
│  - Groups by product_group_id                                                │
│  - Keeps only the highest-scoring item per group                             │
│  - Slices to final limit                                                     │
│                                                                              │
│  Why client-side?                                                            │
│  - Simple Python loop: ~1ms vs ~150ms in SQL                                 │
│  - No database load for grouping operations                                  │
│  - Flexibility to change dedup logic without schema migration                │
│                                                                              │
│  Output: Deduplicated list of unique products                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Complete Flow Example:**

```python
# Step 1: Call raw SQL RPC (over-fetch 2x)
response = supabase.rpc("search_similar_v4", {
    "query_id": "product-uuid",
    "match_cnt": 24  # Request 24, want 12 final
})
# Returns: 24 results, may include variants (e.g., 3 colors of same sofa)

# Step 2: Client-side dedupe
unique_results = dedupe_by_product_group(response, limit=12)
# Returns: 12 unique products, no color variants
```

### 6. Latency Analysis

**Validated by CTO from US servers: 150-200ms**

| Location | Cold Start | Warm Request | Cached |
|----------|------------|--------------|--------|
| US (near Supabase) | ~300ms | **~150-200ms** ✅ | <50ms |
| India (far from Supabase) | ~1.1s | ~400ms | <50ms |

**Breakdown (from US):**

```
┌─────────────────────────────────────────────────────────┐
│  LATENCY BREAKDOWN (Warm Request from US)               │
├─────────────────────────────────────────────────────────┤
│  Network (TCP + SSL):           ~50ms                   │
│  PostgREST overhead:            ~30ms                   │
│  Database query (HNSW):          ~5ms   ← Raw SQL only  │
│  Client-side dedup:              ~1ms   ← App layer     │
│  Response transfer:             ~15ms                   │
│  ─────────────────────────────────────────────────────  │
│  TOTAL:                        ~100-150ms  ✅           │
└─────────────────────────────────────────────────────────┘
```

**Why this architecture achieves <300ms:**

1. **Database does ONE thing well** - Pure HNSW search with filters (~5ms)
2. **No complex SQL** - No CTEs, no DISTINCT ON, no window functions
3. **Client handles business logic** - Dedup is simple iteration (~1ms)
4. **Over-fetch strategy** - Request 2x, dedupe to final count

**Optimizations Applied:**
1. HNSW index: 500ms → 5ms (100x improvement)
2. Client-side dedup: 150ms → 1ms (moved out of SQL)
3. Session pooling: Reuse TCP connections
4. Response caching: Repeated queries <50ms

**To improve further (if needed):**
- Upgrade compute: Micro → Small (~$10/mo) reduces variance
- Region proximity: Deploy Supabase closer to user base

### 8. API Specification

**Raw SQL RPC (Step 1):**

```
POST /rest/v1/rpc/search_similar_v4

Headers:
  Authorization: Bearer <ANON_KEY>
  apikey: <ANON_KEY>
  Content-Type: application/json

Body:
{
  "query_id": "6a9f9346-5e0c-4011-ba52-d3d95975ad05",
  "match_cnt": 24   // Over-fetch for client-side dedup
}

Response (RAW - may contain variants):
[
  {
    "id": "abc123",
    "name": "Calacatta Gold Marble",
    "product_type": "material",
    "product_group_id": "group-xyz",    // Use this for deduplication
    "image_url": "https://storage.googleapis.com/...",
    "similarity": 0.94
  },
  {
    "id": "def456",
    "name": "Calacatta Gold Marble - Grey",  // Same group, different color
    "product_type": "material",
    "product_group_id": "group-xyz",         // Same product_group_id
    "image_url": "https://storage.googleapis.com/...",
    "similarity": 0.91
  },
  ...
]
```

**After Client-Side Dedup (Step 2):**

```json
[
  {
    "id": "abc123",
    "name": "Calacatta Gold Marble",
    "product_group_id": "group-xyz",
    "similarity": 0.94
  },
  // def456 removed - same product_group_id, lower score
  ...
]
```

### 9. Success Metrics

**Validated by CTO testing from US location.**

| Metric | Target | Achieved | Validation |
|--------|--------|----------|------------|
| Latency (US) | <300ms | **~150-200ms** ✅ | CTO tested via curl |
| Latency (cached) | <100ms | **<50ms** ✅ | Streamlit cache |
| Quality (no variants) | 80% relevant | ✅ | Client-side dedup by product_group_id |
| Embedding coverage | >90% | **97%** (50,543 / 51,891) | Voyage Multimodal-3 |

---

## Image Source Logic

Per CTO specification:

| Product Type | Image Source |
|--------------|--------------|
| material, static | `materialData.files.color_original` |
| paint | `materialData.renderedImage` |
| fixed material, hardware, accessory, not_static | `mesh.rendered_image` |

---

## Quick Start

### Test Similarity Search (curl)

```bash
curl -s -w "\nTotal: %{time_total}s\n" -X POST \
  "https://glfevldtqujajsalahxd.supabase.co/rest/v1/rpc/search_similar_v4" \
  -H "Authorization: Bearer <ANON_KEY>" \
  -H "apikey: <ANON_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"query_id": "6a9f9346-5e0c-4011-ba52-d3d95975ad05", "match_cnt": 24}'
```

### Run Streamlit Demo

```bash
cd mtbrd
source venv/bin/activate
streamlit run app.py
```

---

## Files

```
mtbrd/
├── app.py                          # Streamlit demo UI
├── scripts/
│   ├── config.py                   # Configuration
│   ├── generate_all_embeddings.py  # Embedding pipeline
│   └── upload_embeddings.py        # Push to Supabase
├── data/                           # Generated data (gitignored)
└── requirements.txt
```
