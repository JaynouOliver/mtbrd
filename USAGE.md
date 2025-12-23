# Usage Guide

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt

# Additional dependencies for APIs (if not in requirements.txt)
pip install fastapi uvicorn psycopg2-binary asyncpg python-dotenv
```

### 2. Environment Variables

Create a `.env` file in the project root with your database credentials:

```env
user=your_db_user
password=your_db_password
host=your_db_host
port=5432
dbname=your_db_name
```

---

## Cron Jobs

### SQL Cron - Generate Embeddings

**File:** `cron/sql_cron.py`

**Description:** Generates Voyage embeddings for products missing embeddings in products_search table. Finds products without embeddings and updates them.

**Run:**
```bash
source venv/bin/activate
python cron/sql_cron.py
```

**Note:** Set `UPDATE_TABLE = True` in the script to actually update the database. Set `PROCESS_LIMIT` to limit the number of products processed.

---

## API Endpoints

### Products API - Basic Read Operations

**File:** `api/products_api.py`

**Description:** API endpoint for basic read info of records from `products_search` table. Supports pagination, filtering, sorting, and search.

**Run:**
```bash
source venv/bin/activate
python api/products_api.py
# or
uvicorn api.products_api:app --host 0.0.0.0 --port 8004 --reload
```

**Base URL:** `http://localhost:8004`

#### Health Check

```bash
curl -X GET "http://localhost:8004/health"
```

#### Get Single Product by ID

```bash
# Basic product retrieval
curl -X GET "http://localhost:8004/products/6a9f9346-5e0c-4011-ba52-d3d95975ad05"

# With embeddings (large payload)
curl -X GET "http://localhost:8004/products/6a9f9346-5e0c-4011-ba52-d3d95975ad05?include_embeddings=true"
```

#### Get Product Details (Alias)

```bash
# Same as GET /products/{id}
curl -X GET "http://localhost:8004/products/6a9f9346-5e0c-4011-ba52-d3d95975ad05/details"

# With embeddings
curl -X GET "http://localhost:8004/products/6a9f9346-5e0c-4011-ba52-d3d95975ad05/details?include_embeddings=true"
```

#### List Products - Basic Pagination

```bash
# First page, default page size (20)
curl -X GET "http://localhost:8004/products?page=1"

# Second page with custom page size
curl -X GET "http://localhost:8004/products?page=2&page_size=10"

# Small page size for testing
curl -X GET "http://localhost:8004/products?page=1&page_size=5"
```

#### List Products - Filtering

```bash
# Filter by productType
curl -X GET "http://localhost:8004/products?productType=material&page=1&page_size=20"

# Filter by objectStatus
curl -X GET "http://localhost:8004/products?objectStatus=APPROVED&page=1"

# Filter by productType AND objectStatus
curl -X GET "http://localhost:8004/products?productType=material&objectStatus=APPROVED_PRO&page=1"

# Filter by supplier
curl -X GET "http://localhost:8004/products?supplier=Behr&page=1"

# Filter by product_group_id
curl -X GET "http://localhost:8004/products?product_group_id=some-group-uuid&page=1"
```

#### List Products - Search

```bash
# Search by name (case-insensitive)
curl -X GET "http://localhost:8004/products?search=wood&page=1"

# Search with other filters
curl -X GET "http://localhost:8004/products?search=blue&productType=material&page=1"
```

#### List Products - Sorting

```bash
# Sort by name ascending
curl -X GET "http://localhost:8004/products?sort_by=name&sort_order=asc&page=1"

# Sort by name descending
curl -X GET "http://localhost:8004/products?sort_by=name&sort_order=desc&page=1"

# Sort by updatedAt descending
curl -X GET "http://localhost:8004/products?sort_by=updatedAt&sort_order=desc&page=1"

# Sort by createdAt ascending
curl -X GET "http://localhost:8004/products?sort_by=createdAt&sort_order=asc&page=1"
```

#### Combined Filters, Search, and Sorting

```bash
# Filter + search + sort
curl -X GET "http://localhost:8004/products?productType=material&search=wood&sort_by=name&sort_order=asc&page=1&page_size=10"

# Multiple filters + sort
curl -X GET "http://localhost:8004/products?productType=material&objectStatus=APPROVED&sort_by=updatedAt&sort_order=desc&page=1&page_size=20"
```

#### Pretty Print JSON (with jq)

```bash
# Health check
curl -X GET "http://localhost:8004/health" | jq

# Single product
curl -X GET "http://localhost:8004/products/6a9f9346-5e0c-4011-ba52-d3d95975ad05" | jq

# List products
curl -X GET "http://localhost:8004/products?page=1&page_size=2" | jq
```

#### Get a Valid Product ID for Testing

```bash
# Get first product ID from the list
curl -X GET "http://localhost:8004/products?page=1&page_size=1" | jq '.products[0].id'
```

#### Verbose Output (See Headers and Timing)

```bash
# See request/response headers
curl -v -X GET "http://localhost:8004/products?page=1"

# See timing information
curl -w "\n\nTime: %{time_total}s\n" -X GET "http://localhost:8004/products?page=1"
```

---

### Similarity Search

**File:** `api/local_latency_api.py`

**Description:** Low-latency similarity search API using Voyage and DINOv2 embeddings. Optimized for <300ms response times.

**Run:**
```bash
source venv/bin/activate
uvicorn api.local_latency_api:app --host 0.0.0.0 --port 8010 --reload
```

**Base URL:** `http://localhost:8010`

**Endpoints:**
- `GET /voyage?id={product_id}&limit=10` - Voyage similarity search
- `GET /dinov2?id={product_id}&limit=10` - DINOv2 similarity search
- `GET /health` - Health check

**Example:**
```bash
curl -X GET "http://localhost:8010/voyage?id=0001b8a9-c531-4151-90f9-8c07b47d4e7d&limit=10"
curl -X GET "http://localhost:8010/dinov2?id=0001b8a9-c531-4151-90f9-8c07b47d4e7d&limit=10"
```

**Sample Response (Voyage):**
```json
[
    {
        "id": "Material-DfqRFXeBq6",
        "similarity_score": 0.9593,
        "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-DfqRFXeBq6/2174019_diff.jpg"
    },
    {
        "id": "Material-07258",
        "similarity_score": 0.9234,
        "thumbnail_url": "https://storage.googleapis.com/mattoboard-staging.appspot.com/gltf-materials/Material-kst3hs7840/Material-kst3hs7840_color.jpg"
    }
]
```

**Average Response Time:** 120-240ms

---

### SAM3 Server - Image Segmentation

**File:** `api/sam3_server.py`

**Description:** SAM3 image segmentation and vector search API. Segments images and performs similarity search on each segment.

**Run:**
```bash
source venv/bin/activate
uvicorn api.sam3_server:app --host 0.0.0.0 --port 8001 --reload
```

**Base URL:** `http://localhost:8001`

**Endpoint:** `POST /search/image`

**Request Body:**
```json
{
    "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bW9kZXJuJTIwaW50ZXJpb3J8ZW58MHx8MHx8fDA%3D",
    "limit": 10,
    "similarity_threshold": 0.5
}
```

**Request with Optional Filters:**
```json
{
    "image_url": "https://crayonhome.com/wp-content/uploads/2023/06/living-room-interior-wall-mockup-warm-tones-with-pink-armchair-minimal-design-3d-rendering.jpg",
    "limit": 5,
    "similarity_threshold": 0.4,
    "application": "furniture",
    "region_served": ["North America"],
    "relative_price": 4,
    "sustainability_and_health": ["eco-friendly"]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/search/image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bW9kZXJuJTIwaW50ZXJpb3J8ZW58MHx8MHx8fDA%3D",
    "limit": 10,
    "similarity_threshold": 0.5
  }'
```

**Sample Response:**
```json
{
    "segments_processed": 5,
    "segment_results": [
        {
            "segment_index": 0,
            "embedding_dimensions": 1024,
            "count": 5,
            "results": [
                {
                    "id": "Accessory-02105",
                    "similarity": 0.5758,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-02105/3i8oxo70wz_render.jpg"
                },
                {
                    "id": "Material-07258",
                    "similarity": 0.5733,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-staging.appspot.com/gltf-materials/Material-kst3hs7840/Material-kst3hs7840_color.jpg"
                },
                {
                    "id": "Accessory-02109",
                    "similarity": 0.5661,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-02109/dxrq8t715e_render.jpg"
                },
                {
                    "id": "Accessory-02104",
                    "similarity": 0.5493,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-02104/4a8dvougw3_render.jpg"
                },
                {
                    "id": "Accessory-02106",
                    "similarity": 0.5412,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-02106/n2bxc3uflg_render.jpg"
                }
            ]
        },
        {
            "segment_index": 1,
            "embedding_dimensions": 1024,
            "count": 0,
            "results": []
        },
        {
            "segment_index": 2,
            "embedding_dimensions": 1024,
            "count": 5,
            "results": [
                {
                    "id": "89fcb392-b7a2-4077-9b61-799f0438be7e",
                    "similarity": 0.4498,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-89fcb392-b7a2-4077-9b61-799f0438be7e_color_original_4591af7e-19dd-430a-b178-2ccc3aafb8b7.jpg"
                },
                {
                    "id": "d49588d5-b4b9-426b-ac05-4d89ef9d0589",
                    "similarity": 0.439,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-d49588d5-b4b9-426b-ac05-4d89ef9d0589_color_original_7dec36a0-0f41-4003-92f3-688399b6db23.jpg"
                },
                {
                    "id": "Material-BMEMhl21wp",
                    "similarity": 0.4334,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-BMEMhl21wp/10101341_diff.jpg"
                },
                {
                    "id": "Material-06829",
                    "similarity": 0.4308,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-e9akd6fgvq/Material-e9akd6fgvq_color.jpg"
                },
                {
                    "id": "505331bd-b9f0-4a4c-9f93-2451891be023",
                    "similarity": 0.4152,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-505331bd-b9f0-4a4c-9f93-2451891be023_color_original_2c52b8c1-d6fa-446f-8c3b-5456ab101773.jpg"
                }
            ]
        },
        {
            "segment_index": 3,
            "embedding_dimensions": 1024,
            "count": 0,
            "results": []
        },
        {
            "segment_index": 4,
            "embedding_dimensions": 1024,
            "count": 5,
            "results": [
                {
                    "id": "Material-00399",
                    "similarity": 0.6864,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-00399/sak6h1hmre_color.jpg"
                },
                {
                    "id": "Accessory-01282",
                    "similarity": 0.6619,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-01282/lpup181ojd_render.jpg"
                },
                {
                    "id": "1379edf6-df5a-4f70-a2bf-40035a1b1eed",
                    "similarity": 0.6601,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-1379edf6-df5a-4f70-a2bf-40035a1b1eed_color_original_d9a826e8-47ed-4838-90ba-5a897a783d2d.jpg"
                },
                {
                    "id": "242b3a0c-0281-4e1e-821f-eb8b5d473eb6",
                    "similarity": 0.6556,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-242b3a0c-0281-4e1e-821f-eb8b5d473eb6_color_original_45a95c86-440b-411d-bd02-07784dd2fdc7.jpg"
                },
                {
                    "id": "Accessory-01277",
                    "similarity": 0.6522,
                    "thumbnail_url": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Accessory-01277/ez6o6hhy3u_render.jpg"
                }
            ]
        }
    ]
}
```

**Average Response Time:** 11-24 seconds

---

## Notes

- All APIs use connection pooling for optimal performance
- Embeddings are excluded by default from responses (they're large)
- Use `include_embeddings=true` query parameter to include embedding vectors
- All endpoints return raw dictionary responses (no Pydantic validation)
- CORS is enabled for all origins (configure as needed for production)

