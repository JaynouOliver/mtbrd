#!/usr/bin/env python3
"""
Quick inference test for the fine-tuned Real-ESRGAN model.

Usage:
    python test_inference.py --input test_images/ --output results/
    python test_inference.py --input single_image.jpg --output upscaled.jpg
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, scale: int = 2, gpu_id: int = 0):
    """Load the fine-tuned Real-ESRGAN model."""
    
    # Network architecture (must match training config)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )
    
    # Create upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,  # 0 for no tiling, use for large images
        tile_pad=10,
        pre_pad=0,
        half=True,  # FP16 for faster inference
        gpu_id=gpu_id
    )
    
    return upsampler


def upscale_image(upsampler, image_path: Path, output_path: Path) -> dict:
    """Upscale a single image."""
    start_time = time.time()
    
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"success": False, "error": "Failed to read image"}
    
    # Get original size
    h, w = img.shape[:2]
    
    try:
        # Upscale
        output, _ = upsampler.enhance(img, outscale=2)
        
        # Save result
        cv2.imwrite(str(output_path), output)
        
        elapsed = time.time() - start_time
        new_h, new_w = output.shape[:2]
        
        return {
            "success": True,
            "input_size": f"{w}x{h}",
            "output_size": f"{new_w}x{new_h}",
            "time_ms": elapsed * 1000
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test Real-ESRGAN inference")
    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", required=True, help="Output image or folder")
    parser.add_argument("--model", "-m", default=None, help="Model path (default: latest checkpoint)")
    parser.add_argument("--scale", "-s", type=int, default=2, help="Upscale factor")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        # Look for latest checkpoint
        exp_dir = Path(__file__).parent / "experiments" / "RealESRGAN_x2plus_Topaz_fast" / "models"
        if exp_dir.exists():
            checkpoints = list(exp_dir.glob("net_g_*.pth"))
            if checkpoints:
                model_path = str(sorted(checkpoints)[-1])
                logger.info(f"Using checkpoint: {model_path}")
            else:
                # Use pretrained as fallback
                model_path = str(Path(__file__).parent / "experiments" / "pretrained_models" / "RealESRGAN_x2plus.pth")
        else:
            model_path = str(Path(__file__).parent / "experiments" / "pretrained_models" / "RealESRGAN_x2plus.pth")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Download pretrained model first:")
        logger.info("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    upsampler = load_model(model_path, scale=args.scale, gpu_id=args.gpu)
    logger.info("Model loaded successfully")
    
    # Process input
    if input_path.is_file():
        # Single image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = upscale_image(upsampler, input_path, output_path)
        
        if result["success"]:
            logger.info(f"Upscaled: {input_path.name}")
            logger.info(f"  Input size: {result['input_size']}")
            logger.info(f"  Output size: {result['output_size']}")
            logger.info(f"  Time: {result['time_ms']:.1f}ms")
            logger.info(f"  Saved to: {output_path}")
        else:
            logger.error(f"Failed: {result['error']}")
    
    elif input_path.is_dir():
        # Folder of images
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(images)} images to process")
        
        total_time = 0
        successful = 0
        
        for img_path in images:
            out_path = output_path / f"{img_path.stem}_upscaled{img_path.suffix}"
            result = upscale_image(upsampler, img_path, out_path)
            
            if result["success"]:
                successful += 1
                total_time += result["time_ms"]
                logger.info(f"[{successful}/{len(images)}] {img_path.name}: {result['time_ms']:.1f}ms")
            else:
                logger.error(f"Failed {img_path.name}: {result['error']}")
        
        avg_time = total_time / successful if successful > 0 else 0
        logger.info("=" * 60)
        logger.info("INFERENCE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {successful}/{len(images)} images")
        logger.info(f"Average time: {avg_time:.1f}ms per image")
        logger.info(f"Throughput: {1000/avg_time:.1f} images/second" if avg_time > 0 else "N/A")
        logger.info(f"Results saved to: {output_path}")
    
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()


