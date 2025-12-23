#!/usr/bin/env python3
"""
Products API - Basic read operations for products_search table.

Endpoints:
- GET /products/{id} - Get single product by ID
- GET /products - List products with pagination, filtering, and sorting
- GET /products/{id}/details - Get full product details (all fields)

Supports filtering by:
- productType (material, paint, etc.)
- objectStatus (APPROVED, APPROVED_PRO, etc.)
- supplier
- product_group_id
- Search by name (text search)

Supports pagination and sorting.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager, contextmanager
import psycopg2
from psycopg2 import pool
import psycopg2.extras
import threading
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection config
DB_CONFIG = {
    "user": os.getenv("user"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port", "5432"),
    "database": os.getenv("dbname", "postgres"),
}

# Global connection pool
db_pool: pool.ThreadedConnectionPool = None


def get_db_connection():
    """
    Establish and return a PostgreSQL database connection using environment variables.
    Same pattern as database/Direct_connection.py
    """
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        return connection
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise


@contextmanager
def get_db_connection_from_pool():
    """Get a connection from the pool with automatic return."""
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    finally:
        if conn:
            db_pool.putconn(conn)


def keepalive_task():
    """Background task to keep connections warm."""
    while True:
        time.sleep(30)
        try:
            with get_db_connection_from_pool() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception:
            pass


keepalive_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - initialize connection pool at startup."""
    global db_pool, keepalive_thread
    
    print("Initializing psycopg2 connection pool for Products API...")
    
    # Create connection pool
    db_pool = pool.ThreadedConnectionPool(
        minconn=3,
        maxconn=15,
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"]
    )
    
    print(f"Pool initialized (min: 3, max: 15)")
    
    # Start keepalive thread
    keepalive_thread = threading.Thread(target=keepalive_task, daemon=True)
    keepalive_thread.start()
    
    yield
    
    # Cleanup on shutdown
    if db_pool:
        db_pool.closeall()
    print("Connection pool closed")


app = FastAPI(
    title="Products API",
    description="Basic read operations for products_search table",
    version="1.0.0",
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




def get_thumbnail_url(row: Dict[str, Any]) -> Optional[str]:
    """Extract thumbnail URL from various possible locations."""
    # Try materialData first
    if row.get("materialData"):
        material_data = row["materialData"]
        if isinstance(material_data, dict):
            files = material_data.get("files")
            if isinstance(files, dict):
                url = files.get("color_original")
                if url:
                    return url
    
    # Try mesh
    if row.get("mesh"):
        mesh = row["mesh"]
        if isinstance(mesh, dict):
            url = mesh.get("rendered_image")
            if url:
                return url
    
    # Try metadata
    if row.get("metadata"):
        metadata = row["metadata"]
        if isinstance(metadata, dict):
            material_data = metadata.get("materialData")
            if isinstance(material_data, dict):
                files = material_data.get("files")
                if isinstance(files, dict):
                    url = files.get("color_original")
                    if url:
                        return url
    
    return None


@app.get("/health")
def health_check():
    """Health check with connection test."""
    start = time.perf_counter()
    try:
        with get_db_connection_from_pool() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        connection_time = (time.perf_counter() - start) * 1000
        return {
            "status": "healthy",
            "db_connected": True,
            "pool_size": db_pool.maxconn if db_pool else 0,
            "connection_time_ms": round(connection_time, 2)
        }
    except Exception as e:
        return {
            "status": f"unhealthy: {str(e)}",
            "db_connected": False,
            "pool_size": 0,
            "connection_time_ms": 0
        }


@app.get("/products/{product_id}")
def get_product(
    product_id: str,
    include_embeddings: bool = Query(False, description="Include embedding vectors (large payload)")
):
    """
    Get a single product by ID with full details.
    
    Args:
        product_id: Product UUID
        include_embeddings: If True, include embedding_visual and embedding_dinov2 vectors
    
    Returns:
        Full product details as raw dictionary
    """
    try:
        with get_db_connection_from_pool() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Use SELECT * - PostgreSQL handles column names correctly
            # If embeddings not needed, we'll filter them out in Python
            query = "SELECT * FROM products_search WHERE id = %s"
            
            cursor.execute(query, (product_id,))
            row = cursor.fetchone()
            cursor.close()
            
            if not row:
                raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
            
            # Convert to dict
            product_dict = dict(row)
            
            # Remove embedding vectors if not requested (they're large)
            if not include_embeddings:
                product_dict.pop("embedding_visual", None)
                product_dict.pop("embedding_dinov2", None)
                product_dict.pop("embedding_semantic", None)
            
            # Add thumbnail_url
            product_dict["thumbnail_url"] = get_thumbnail_url(product_dict)
            
            return product_dict
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products")
def list_products(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    productType: Optional[str] = Query(None, description="Filter by productType (e.g., 'material', 'paint')"),
    objectStatus: Optional[str] = Query(None, description="Filter by objectStatus (e.g., 'APPROVED', 'APPROVED_PRO')"),
    supplier: Optional[str] = Query(None, description="Filter by supplier name"),
    product_group_id: Optional[str] = Query(None, description="Filter by product_group_id"),
    search: Optional[str] = Query(None, description="Search in product name (case-insensitive partial match)"),
    sort_by: str = Query("id", description="Sort field (id, name, updatedAt, createdAt)"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order")
):
    """
    List products with pagination, filtering, and sorting.
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page (1-100)
        productType: Filter by productType
        objectStatus: Filter by objectStatus
        supplier: Filter by supplier
        product_group_id: Filter by product_group_id
        search: Search term for product name
        sort_by: Field to sort by (id, name, updatedAt, createdAt)
        sort_order: Sort direction (asc or desc)
    
    Returns:
        Paginated list of products as raw dictionaries
    """
    try:
        with get_db_connection_from_pool() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build WHERE clause
            where_conditions = []
            params = []
            
            if productType:
                where_conditions.append('"productType" = %s')
                params.append(productType)
            
            if objectStatus:
                where_conditions.append('"objectStatus" = %s')
                params.append(objectStatus)
            
            if supplier:
                where_conditions.append("supplier = %s")
                params.append(supplier)
            
            if product_group_id:
                where_conditions.append("product_group_id = %s")
                params.append(product_group_id)
            
            if search:
                where_conditions.append("name ILIKE %s")
                params.append(f"%{search}%")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Validate sort_by field and quote camelCase columns
            valid_sort_fields = ["id", "name", "updatedAt", "createdAt"]
            if sort_by not in valid_sort_fields:
                sort_by = "id"
            
            # Quote camelCase columns for sorting
            camel_case_fields = ["updatedAt", "createdAt", "productType", "objectStatus"]
            sort_field = f'"{sort_by}"' if sort_by in camel_case_fields else sort_by
            sort_clause = f"ORDER BY {sort_field} {sort_order.upper()}"
            
            # Count total
            count_query = f"SELECT COUNT(*) as count FROM products_search WHERE {where_clause}"
            cursor.execute(count_query, params)
            count_result = cursor.fetchone()
            total = count_result['count'] if count_result else 0
            
            # Calculate pagination
            offset = (page - 1) * page_size
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0
            
            # Fetch products - use SELECT * to avoid column name issues
            query = f"""
                SELECT * FROM products_search
                WHERE {where_clause}
                {sort_clause}
                LIMIT %s OFFSET %s
            """
            params.extend([page_size, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Build response - convert rows to dicts and add thumbnail_url
            products = []
            for row in rows:
                product_dict = dict(row)
                # Remove embedding vectors (they're large and not needed for list view)
                product_dict.pop("embedding_visual", None)
                product_dict.pop("embedding_dinov2", None)
                product_dict.pop("embedding_semantic", None)
                # Add thumbnail_url
                product_dict["thumbnail_url"] = get_thumbnail_url(product_dict)
                products.append(product_dict)
            
            cursor.close()
            
            return {
                "products": products,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error in list_products: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/products/{product_id}/details")
def get_product_details(
    product_id: str,
    include_embeddings: bool = Query(False, description="Include embedding vectors (large payload)")
):
    """
    Get full product details (alias for GET /products/{id}).
    Included for API clarity - same as GET /products/{id}.
    """
    return get_product(product_id, include_embeddings)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
