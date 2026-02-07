"""
Real-ESRGAN Modal Deployment (Optimized Version)
=================================================
Optimizations:
1. Keep model in GPU memory (no subprocess, direct import)
2. Default to 2x model (faster)
6. Tiling for large images (reduces VRAM, more stable)

Deploy:
    modal deploy modal_esrgan_optimized.py

Test:
    curl -F "file=@image.jpg" https://<your-url>/upscale
"""

import modal
import io
import time
import tempfile
import os
import uuid

# ============================================
# Modal App Configuration  
# ============================================

app = modal.App("esrgan-api-optimized")  # Different app name!

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "basicsr",
        "facexlib",
        "gfpgan",
        "google-cloud-storage",
    )
    .run_commands(
        # Clone Real-ESRGAN and install (this worked before!)
        "git clone https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN",
        "cd /opt/Real-ESRGAN && python setup.py develop",
        # Fix basicsr compatibility with new torchvision
        "grep -rl 'functional_tensor' /usr/local/lib/python3.10/site-packages/basicsr/ | xargs -r sed -i 's/functional_tensor/functional/g'",
        # Clear bytecode cache
        "find /usr/local/lib/python3.10/site-packages/basicsr/ -name '*.pyc' -delete || true",
        "find /usr/local/lib/python3.10/site-packages/basicsr/ -name '__pycache__' -type d -exec rm -rf {} + || true",
    )
    .run_commands(
        "mkdir -p /opt/Real-ESRGAN/weights",
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /opt/Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /opt/Real-ESRGAN/weights/RealESRGAN_x2plus.pth",
    )
)

# ============================================
# GCS Configuration (using Modal Secrets)
# ============================================

GCS_BUCKET = "mattoboard-staging.appspot.com"
GCS_FOLDER = "esrgan_upscaled"


def get_gcs_credentials():
    """Load GCS credentials from Modal secret."""
    import os
    import json
    
    creds_json = os.environ.get("GCS_CREDENTIALS")
    if not creds_json:
        raise ValueError("GCS_CREDENTIALS secret not found. Please create it in Modal.")
    
    return json.loads(creds_json)


def upload_to_gcs(image_bytes: bytes, content_type: str = "image/png", extension: str = "png") -> str:
    """Upload image bytes to GCS and return public URL."""
    from google.cloud import storage
    from google.oauth2 import service_account
    
    short_id = uuid.uuid4().hex[:12]
    blob_name = f"{GCS_FOLDER}/upscaled_{short_id}.{extension}"
    
    creds = get_gcs_credentials()
    credentials = service_account.Credentials.from_service_account_info(creds)
    client = storage.Client(credentials=credentials, project=creds["project_id"])
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    
    blob.cache_control = "public,max-age=31536000"
    blob.upload_from_string(image_bytes, content_type=content_type)
    blob.make_public()
    
    public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{blob_name}"
    return public_url


MAX_FILE_SIZE_KB = 700


def compress_image_to_limit(img, max_size_kb: int = MAX_FILE_SIZE_KB) -> tuple:
    """
    Compress image to fit under max_size_kb.
    Returns (image_bytes, content_type, extension, was_compressed).
    """
    from PIL import Image
    import cv2
    
    max_bytes = max_size_kb * 1024
    
    # First, try PNG encoding
    _, buf = cv2.imencode(".png", img)
    if len(buf) <= max_bytes:
        return buf.tobytes(), "image/png", "png", False
    
    print(f"  Image is {len(buf) / 1024:.0f} KB (limit {max_size_kb} KB), compressing...")
    
    # Convert to PIL for compression
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Phase 1: try quality reduction
    for quality in (95, 85, 75, 65):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            print(f"  Compressed to {buf.tell() / 1024:.0f} KB at quality={quality}")
            return buf.getvalue(), "image/jpeg", "jpg", True
    
    # Phase 2: progressively scale down
    scale = 0.9
    while scale > 0.1:
        new_w = max(int(pil_img.width * scale), 1)
        new_h = max(int(pil_img.height * scale), 1)
        if min(new_w, new_h) < 100:
            break
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=75, optimize=True)
        if buf.tell() <= max_bytes:
            print(f"  Compressed to {buf.tell() / 1024:.0f} KB at {new_w}x{new_h}")
            return buf.getvalue(), "image/jpeg", "jpg", True
        scale -= 0.1
    
    print(f"  WARNING: Could not compress below {max_size_kb} KB")
    return buf.getvalue(), "image/jpeg", "jpg", True


# ============================================
# Model Cache (OPTIMIZATION 1: Keep in memory)
# ============================================

MODEL_CACHE = {}

MODEL_CONFIGS = {
    "RealESRGAN_x2plus": {
        "scale": 2,
        "path": "/opt/Real-ESRGAN/weights/RealESRGAN_x2plus.pth",
        "num_block": 23,
    },
    "RealESRGAN_x4plus": {
        "scale": 4,
        "path": "/opt/Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
        "num_block": 23,
    }
}


def get_upsampler(model_name: str, tile: int = 256):
    """Get or create upsampler (cached in GPU memory)."""
    cache_key = f"{model_name}_{tile}"
    
    if cache_key not in MODEL_CACHE:
        print(f"🔄 Loading {model_name} (tile={tile})...")
        
        import sys
        sys.path.insert(0, "/opt/Real-ESRGAN")
        
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        config = MODEL_CONFIGS[model_name]
        
        # Create network architecture
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=config["num_block"],
            num_grow_ch=32,
            scale=config["scale"]
        )
        
        # Create upsampler with tiling (OPTIMIZATION 6)
        MODEL_CACHE[cache_key] = RealESRGANer(
            scale=config["scale"],
            model_path=config["path"],
            model=net,
            tile=tile,  # Tiling for large images
            tile_pad=10,
            pre_pad=0,
            half=True,  # FP16 for speed
        )
        print(f"✅ {model_name} loaded!")
    
    return MODEL_CACHE[cache_key]


# ============================================
# FastAPI App
# ============================================

from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import JSONResponse

fastapi_app = FastAPI(title="Real-ESRGAN API (Optimized)")


@fastapi_app.get("/")
def health():
    return {"status": "healthy", "service": "Real-ESRGAN", "optimized": True}


@fastapi_app.get("/models")
def list_models():
    return {
        "available": list(MODEL_CONFIGS.keys()),
        "default": "RealESRGAN_x2plus",  # OPTIMIZATION 2: Default to 2x
        "loaded": list(MODEL_CACHE.keys()),
    }


@fastapi_app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model: str = Form("RealESRGAN_x2plus"),  # OPTIMIZATION 2: Default to 2x
    scale: float = Form(None),  # Auto-detect from model if not specified
    tile: int = Form(256),  # OPTIMIZATION 6: Tiling (0=off, 256=default)
):
    """
    Upscale image using Real-ESRGAN.
    
    Args:
        file: Image file to upscale
        model: Model name (RealESRGAN_x2plus, RealESRGAN_x4plus)
        scale: Output scale (auto-detected from model if not set)
        tile: Tile size for processing (0=no tiling, 256=default)
    
    Returns:
        JSON with public URL and metadata
    """
    import cv2
    import numpy as np
    
    try:
        # Validate model
        if model not in MODEL_CONFIGS:
            return JSONResponse(
                {"error": f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}"},
                status_code=400
            )
        
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)
        
        h, w = img.shape[:2]
        print(f"📐 Input: {w}x{h}")
        
        # Get upsampler (cached!)
        upsampler = get_upsampler(model, tile)
        
        # Determine output scale
        model_scale = MODEL_CONFIGS[model]["scale"]
        outscale = scale if scale else model_scale
        
        # Run inference
        start = time.time()
        print(f"🚀 Upscaling with {model} (tile={tile}, outscale={outscale})...")
        
        output, _ = upsampler.enhance(img, outscale=outscale)
        
        inference_ms = int((time.time() - start) * 1000)
        out_h, out_w = output.shape[:2]
        print(f"⏱️ Inference: {inference_ms}ms")
        print(f"📐 Output: {out_w}x{out_h}")
        
        # Compress if needed
        image_bytes, content_type, extension, was_compressed = compress_image_to_limit(output)
        file_size_kb = len(image_bytes) / 1024
        print(f"📦 Final size: {file_size_kb:.0f} KB (compressed: {was_compressed})")
        
        # Upload to GCS
        print("📤 Uploading to GCS...")
        upload_start = time.time()
        public_url = upload_to_gcs(image_bytes, content_type, extension)
        upload_ms = int((time.time() - upload_start) * 1000)
        print(f"✅ Uploaded in {upload_ms}ms")
        
        total_ms = int((time.time() - start) * 1000)
        
        return {
            "status": "success",
            "url": public_url,
            "input_size": f"{w}x{h}",
            "output_size": f"{out_w}x{out_h}",
            "scale": outscale,
            "model": model,
            "tile": tile,
            "format": extension,
            "file_size_kb": round(file_size_kb, 1),
            "compressed": was_compressed,
            "inference_time_ms": inference_ms,
            "upload_time_ms": upload_ms,
            "total_time_ms": total_ms,
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================
# Modal Entry Point
# ============================================

@app.function(
    image=gpu_image, 
    gpu="L4", 
    timeout=300, 
    min_containers=1, 
    scaledown_window=120,
    secrets=[modal.Secret.from_name("gcs-credentials")]
)
@modal.asgi_app()
def web():
    return fastapi_app


@app.local_entrypoint()
def main():
    print("🚀 Real-ESRGAN API (Optimized)")
    print("Deploy: modal deploy modal_esrgan_optimized.py")
