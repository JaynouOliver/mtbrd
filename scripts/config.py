"""
Configuration for embedding pipeline.
Set environment variables before running scripts.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_FILE = DATA_DIR / "products_metadata.json"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Voyage AI
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

# Image source logic based on CTO's notes
# productType -> (json_path_to_image, is_upscaled)
IMAGE_SOURCE_MAP = {
    # Materials: use color_original (Topaz upscaled)
    "material": ("materialData.files.color_original", True),
    "static": ("materialData.files.color_original", True),
    
    # Paint: use renderedImage
    "paint": ("materialData.renderedImage", False),
    
    # Products: use mesh.rendered_image
    "fixed material": ("mesh.rendered_image", False),
    "hardware": ("mesh.rendered_image", False),
    "accessory": ("mesh.rendered_image", False),
    "not_static": ("mesh.rendered_image", False),
    "paintObject": ("mesh.rendered_image", False),
}

# Product types to process
VALID_PRODUCT_TYPES = list(IMAGE_SOURCE_MAP.keys())

# Valid object statuses
VALID_STATUSES = ["APPROVED", "APPROVED_PRO"]

# Batch sizes
EXPORT_BATCH_SIZE = 1000
DOWNLOAD_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE = 20  # Voyage rate limits
DB_PUSH_BATCH_SIZE = 100

# Embedding model
VOYAGE_MODEL = "voyage-multimodal-3.5"
EMBEDDING_DIMENSION = 1024  # voyage-multimodal-3.5 outputs 1024 dimensions

# Rate limits (voyage-multimodal-3.5) - upgraded tier
RATE_LIMIT_TPM = 10000  # Tokens per minute
RATE_LIMIT_RPM = 4000   # Requests per minute (upgraded from 3)


def get_image_url_from_product(product: dict) -> str | None:
    """
    Extract the correct image URL based on product type.
    Uses CTO's logic for image source selection.
    """
    product_type = product.get("productType")
    
    if product_type not in IMAGE_SOURCE_MAP:
        return None
    
    json_path, _ = IMAGE_SOURCE_MAP[product_type]
    
    # Navigate the nested JSON path
    parts = json_path.split(".")
    value = product
    
    for part in parts:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    
    return value if isinstance(value, str) else None


def validate_config():
    """Validate that required environment variables are set."""
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

