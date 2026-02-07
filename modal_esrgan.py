"""
Real-ESRGAN Modal Deployment (Clean Version)
=============================================
Uses the official Real-ESRGAN inference script directly.
Uploads result to GCS and returns a public URL.

Deploy:
    modal deploy modal_esrgan.py

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

app = modal.App("esrgan-api")

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
        "google-cloud-storage",  # For GCS upload
    )
    .run_commands(
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
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -O /opt/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth",
    )
)

# ============================================
# GCS Configuration (Staging)
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
    
    Strategy:
      1. First try reducing JPEG quality (95 -> 85 -> 75 -> 65).
      2. If still too large, scale dimensions down by 10% per step.
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
    
    # Phase 1: try quality reduction at current dimensions
    for quality in (95, 85, 75, 65):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            print(f"  Compressed to {buf.tell() / 1024:.0f} KB at quality={quality} (no resize)")
            return buf.getvalue(), "image/jpeg", "jpg", True
    
    # Phase 2: progressively scale down by 10% and re-encode at quality=75
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
            print(f"  Compressed to {buf.tell() / 1024:.0f} KB at {new_w}x{new_h} (scale={scale:.0%})")
            return buf.getvalue(), "image/jpeg", "jpg", True
        scale -= 0.1
    
    # Fallback: return whatever we have
    print(f"  WARNING: Could not compress below {max_size_kb} KB. Best effort: {buf.tell() / 1024:.0f} KB")
    return buf.getvalue(), "image/jpeg", "jpg", True


# ============================================
# FastAPI App
# ============================================

from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import JSONResponse

fastapi_app = FastAPI(title="Real-ESRGAN API")


@fastapi_app.get("/")
def health():
    return {"status": "healthy", "service": "Real-ESRGAN"}


@fastapi_app.get("/models")
def list_models():
    return {
        "available": ["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B"],
        "default": "RealESRGAN_x4plus"
    }


@fastapi_app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model: str = Form("RealESRGAN_x4plus"),
    scale: float = Form(4.0),
    format: str = Form("png"),
    user_id: str = Form("api_user"),
):
    """
    Upscale image using Real-ESRGAN and return a GCS URL.
    
    Returns JSON with:
        - url: Public URL of the upscaled image
        - input_size: Original dimensions
        - output_size: Upscaled dimensions
        - inference_time_ms: Processing time
    """
    import subprocess
    import cv2
    import numpy as np
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Create temp directory for this request
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.png")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            
            # Save input image
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return JSONResponse({"error": "Failed to decode image"}, status_code=400)
            
            h, w = img.shape[:2]
            cv2.imwrite(input_path, img)
            print(f"📐 Input: {w}x{h}")
            
            # Run inference using the official script
            start = time.time()
            cmd = [
                "python", "/opt/Real-ESRGAN/inference_realesrgan.py",
                "-n", model,
                "-i", input_path,
                "-o", output_dir,
                "--outscale", str(scale),
            ]
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/Real-ESRGAN")
            
            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                return JSONResponse({"error": result.stderr}, status_code=500)
            
            inference_ms = int((time.time() - start) * 1000)
            print(f"⏱️ Inference took {inference_ms}ms")
            
            # Find output file
            output_files = os.listdir(output_dir)
            if not output_files:
                return JSONResponse({"error": "No output generated"}, status_code=500)
            
            output_path = os.path.join(output_dir, output_files[0])
            output_img = cv2.imread(output_path)
            out_h, out_w = output_img.shape[:2]
            print(f"📐 Output: {out_w}x{out_h}")
            
            # Compress if needed (>700KB)
            image_bytes, content_type, extension, was_compressed = compress_image_to_limit(output_img)
            file_size_kb = len(image_bytes) / 1024
            print(f"📦 Final size: {file_size_kb:.0f} KB (compressed: {was_compressed})")
            
            # Upload to GCS
            print("📤 Uploading to GCS...")
            upload_start = time.time()
            public_url = upload_to_gcs(image_bytes, content_type, extension)
            upload_ms = int((time.time() - upload_start) * 1000)
            print(f"✅ Uploaded in {upload_ms}ms: {public_url}")
            
            total_ms = int((time.time() - start) * 1000)
            
            return {
                "status": "success",
                "url": public_url,
                "input_size": f"{w}x{h}",
                "output_size": f"{out_w}x{out_h}",
                "scale": scale,
                "model": model,
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
    print("🚀 Real-ESRGAN API")
    print("Deploy: modal deploy modal_esrgan.py")
