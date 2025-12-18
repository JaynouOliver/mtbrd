#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained upscaler model.
Compares model output against Topaz reference using SSIM, LPIPS, PSNR.

Usage:
    python evaluate_model.py --model path/to/model.pth
    python evaluate_model.py --model path/to/model.pth --samples 100 --output results/
"""

import os
import sys
import argparse
import json
import logging
import random
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import torch
import cv2
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("LPIPS not available. Install with: pip install lpips")


class ModelEvaluator:
    """Evaluates upscaler model against Topaz reference."""
    
    def __init__(self, model_path: str, scale: int = 2, device: str = None):
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Device: {self.device}")
        
        # Create model
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
        
        # Load weights
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "params_ema" in ckpt:
                state = ckpt["params_ema"]
            elif "params" in ckpt:
                state = ckpt["params"]
            else:
                state = ckpt
        else:
            state = ckpt
        
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        
        if self.device == "cuda":
            self.model = self.model.half()
        
        # Initialize LPIPS
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net="alex")
            if self.device == "cuda":
                self.lpips_model = self.lpips_model.cuda()
        
        logger.info("Model loaded successfully")
    
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale an image (RGB uint8 input/output)."""
        h, w = img.shape[:2]
        
        # Pad to even dimensions
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        
        # To tensor
        t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        t = t.unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            t = t.half()
        
        # Inference
        with torch.no_grad():
            out = self.model(t)
        
        # Back to numpy
        out = out.squeeze(0).float().cpu().clamp(0, 1).numpy()
        out = (out.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        
        # Remove padding
        target_h, target_w = h * self.scale, w * self.scale
        out = out[:target_h, :target_w, :]
        
        return out
    
    def compute_metrics(self, pred: np.ndarray, ref: np.ndarray) -> dict:
        """Compute SSIM, PSNR, and LPIPS between prediction and reference."""
        # Ensure same size
        ph, pw = pred.shape[:2]
        rh, rw = ref.shape[:2]
        h, w = min(ph, rh), min(pw, rw)
        
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # SSIM
        ssim_val = ssim(ref, pred, data_range=255, channel_axis=2)
        
        # PSNR
        psnr_val = psnr(ref, pred, data_range=255)
        
        # LPIPS
        lpips_val = None
        if self.lpips_model is not None:
            def to_tensor(im):
                t = torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.0
                t = t.unsqueeze(0) * 2.0 - 1.0
                return t
            
            pred_t = to_tensor(pred)
            ref_t = to_tensor(ref)
            
            if self.device == "cuda":
                pred_t = pred_t.cuda()
                ref_t = ref_t.cuda()
            
            with torch.no_grad():
                lpips_val = self.lpips_model(pred_t, ref_t).item()
        
        return {
            "ssim": ssim_val,
            "psnr": psnr_val,
            "lpips": lpips_val
        }


def fetch_evaluation_pairs(n_fetch: int = 500, n_sample: int = 100) -> list:
    """Fetch random image pairs from Supabase for evaluation."""
    supabase_url = os.environ.get("SUPABASE_URL", "https://glfevldtqujajsalahxd.supabase.co")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    
    if not supabase_key:
        logger.error("SUPABASE_KEY not set")
        return []
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    }
    
    # Fetch more than needed to account for invalid pairs
    url = (
        f"{supabase_url}/rest/v1/productsV2"
        f"?select=id,metadata,materialData"
        f"&productType=in.(material,fixed%20material)"
        f"&objectStatus=in.(APPROVED,APPROVED_PRO)"
        f"&order=updatedAt.desc"
        f"&limit={n_fetch}"
    )
    
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        logger.error(f"Failed to fetch from Supabase: {e}")
        return []
    
    pairs = []
    for item in rows:
        meta = item.get("metadata") or {}
        mat = item.get("materialData") or {}
        files = mat.get("files") or {}
        
        orig = meta.get("materialImageUrl")
        upsc = files.get("color_original")
        
        if orig and upsc and orig != upsc:
            pairs.append({
                "id": item.get("id"),
                "original": orig,
                "topaz": upsc
            })
    
    if len(pairs) < n_sample:
        logger.warning(f"Only found {len(pairs)} valid pairs")
        return pairs
    
    return random.sample(pairs, n_sample)


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as RGB numpy array."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate upscaler model")
    parser.add_argument("--model", "-m", required=True, help="Path to model weights")
    parser.add_argument("--samples", "-n", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", "-o", default="evaluation_results", help="Output directory")
    parser.add_argument("--save-images", action="store_true", help="Save comparison images")
    parser.add_argument("--scale", "-s", type=int, default=2, help="Upscale factor")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, scale=args.scale)
    
    # Fetch evaluation pairs
    logger.info(f"Fetching {args.samples} evaluation pairs from database...")
    pairs = fetch_evaluation_pairs(n_fetch=args.samples * 3, n_sample=args.samples)
    
    if not pairs:
        logger.error("No evaluation pairs available")
        return
    
    logger.info(f"Evaluating on {len(pairs)} image pairs...")
    
    # Evaluation results
    results = []
    ssim_vals, psnr_vals, lpips_vals = [], [], []
    times = []
    failed = 0
    
    for i, pair in enumerate(pairs, 1):
        try:
            logger.info(f"[{i}/{len(pairs)}] Processing {pair['id'][:8]}...")
            
            # Download images
            original = download_image(pair["original"])
            topaz_ref = download_image(pair["topaz"])
            
            # Upscale with our model
            start = time.time()
            predicted = evaluator.upscale(original)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            
            # Compute metrics
            metrics = evaluator.compute_metrics(predicted, topaz_ref)
            
            ssim_vals.append(metrics["ssim"])
            psnr_vals.append(metrics["psnr"])
            if metrics["lpips"] is not None:
                lpips_vals.append(metrics["lpips"])
            
            result = {
                "id": pair["id"],
                "original_size": f"{original.shape[1]}x{original.shape[0]}",
                "output_size": f"{predicted.shape[1]}x{predicted.shape[0]}",
                "ssim": metrics["ssim"],
                "psnr": metrics["psnr"],
                "lpips": metrics["lpips"],
                "time_ms": elapsed_ms
            }
            results.append(result)
            
            logger.info(f"  SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}dB, "
                       f"LPIPS={metrics['lpips']:.4f if metrics['lpips'] else 'N/A'}, "
                       f"Time={elapsed_ms:.1f}ms")
            
            # Save comparison images if requested
            if args.save_images:
                img_dir = output_dir / "images"
                img_dir.mkdir(exist_ok=True)
                
                # Save side-by-side comparison
                h = min(predicted.shape[0], topaz_ref.shape[0])
                w = min(predicted.shape[1], topaz_ref.shape[1])
                pred_crop = predicted[:h, :w]
                ref_crop = topaz_ref[:h, :w]
                
                comparison = np.hstack([pred_crop, ref_crop])
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_dir / f"{pair['id'][:8]}_comparison.jpg"), comparison_bgr)
        
        except Exception as e:
            failed += 1
            logger.error(f"  FAILED: {e}")
    
    # Compute summary statistics
    def stats(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max())
        }
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "total_samples": len(pairs),
        "successful": len(results),
        "failed": failed,
        "metrics": {
            "ssim": stats(ssim_vals),
            "psnr": stats(psnr_vals),
            "lpips": stats(lpips_vals),
            "inference_time_ms": stats(times)
        },
        "targets": {
            "ssim_minimum": 0.90,
            "ssim_target": 0.93,
            "lpips_maximum": 0.10,
            "lpips_target": 0.07
        },
        "results": results
    }
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples: {len(results)} successful, {failed} failed")
    print()
    print("METRICS (comparing to Topaz reference):")
    print("-" * 70)
    print(f"  SSIM:    mean={summary['metrics']['ssim']['mean']:.4f} "
          f"(std={summary['metrics']['ssim']['std']:.4f})")
    print(f"  PSNR:    mean={summary['metrics']['psnr']['mean']:.2f}dB "
          f"(std={summary['metrics']['psnr']['std']:.2f})")
    if summary['metrics']['lpips']['mean'] is not None:
        print(f"  LPIPS:   mean={summary['metrics']['lpips']['mean']:.4f} "
              f"(std={summary['metrics']['lpips']['std']:.4f})")
    print(f"  Time:    mean={summary['metrics']['inference_time_ms']['mean']:.1f}ms "
          f"(std={summary['metrics']['inference_time_ms']['std']:.1f})")
    print()
    print("TARGET COMPARISON:")
    print("-" * 70)
    ssim_pass = summary['metrics']['ssim']['mean'] >= 0.90
    ssim_ideal = summary['metrics']['ssim']['mean'] >= 0.93
    print(f"  SSIM >= 0.90 (minimum):  {'PASS' if ssim_pass else 'FAIL'}")
    print(f"  SSIM >= 0.93 (target):   {'PASS' if ssim_ideal else 'FAIL'}")
    
    if summary['metrics']['lpips']['mean'] is not None:
        lpips_pass = summary['metrics']['lpips']['mean'] <= 0.10
        lpips_ideal = summary['metrics']['lpips']['mean'] <= 0.07
        print(f"  LPIPS <= 0.10 (maximum): {'PASS' if lpips_pass else 'FAIL'}")
        print(f"  LPIPS <= 0.07 (target):  {'PASS' if lpips_ideal else 'FAIL'}")
    
    print()
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    
    # Return exit code based on quality
    if ssim_pass and (summary['metrics']['lpips']['mean'] is None or 
                      summary['metrics']['lpips']['mean'] <= 0.10):
        print("\nOVERALL: PASS - Model meets minimum quality thresholds")
        return 0
    else:
        print("\nOVERALL: FAIL - Model does not meet minimum quality thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
