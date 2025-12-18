#!/usr/bin/env python3
"""
Inference script for the trained Topaz-style upscaler.

Usage:
    python inference.py --input image.jpg --output upscaled.jpg
    python inference.py --input folder/ --output results/
    python inference.py --input image.jpg --output upscaled.jpg --model path/to/model.pth
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TopazUpscaler:
    """Upscaler using the fine-tuned Real-ESRGAN model."""
    
    def __init__(self, model_path: str = None, scale: int = 2, device: str = None):
        """
        Initialize the upscaler.
        
        Args:
            model_path: Path to the trained model. If None, uses latest checkpoint.
            scale: Upscaling factor (default: 2)
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find model
        if model_path is None:
            model_path = self._find_latest_model()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
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
        loadnet = torch.load(model_path, map_location=torch.device("cpu"))
        if "params_ema" in loadnet:
            keyname = "params_ema"
        elif "params" in loadnet:
            keyname = "params"
        else:
            keyname = None
        
        if keyname:
            self.model.load_state_dict(loadnet[keyname], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Use half precision on GPU for speed
        if self.device == "cuda":
            self.model = self.model.half()
        
        logger.info("Model loaded successfully")
    
    def _find_latest_model(self) -> str:
        """Find the latest trained model checkpoint."""
        script_dir = Path(__file__).parent
        
        # Look in common locations
        search_paths = [
            script_dir / "models",
            Path("models"),
            Path("experiments/RealESRGAN_x2_Topaz/models"),
            Path("Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models"),
            Path.home() / "Real-ESRGAN/experiments/RealESRGAN_x2_Topaz/models",
        ]
        
        for path in search_paths:
            if path.exists():
                # Prefer net_g_latest.pth, then highest numbered checkpoint
                latest = path / "net_g_latest.pth"
                if latest.exists():
                    return str(latest)
                
                checkpoints = sorted(path.glob("net_g_*.pth"))
                if checkpoints:
                    return str(checkpoints[-1])
        
        # Fallback to pretrained
        pretrained = Path("experiments/pretrained_models/RealESRGAN_x2plus.pth")
        if pretrained.exists():
            logger.warning("Using pretrained model (no fine-tuned model found)")
            return str(pretrained)
        
        raise FileNotFoundError("No model found. Train a model first or specify --model path")
    
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale an image.
        
        Args:
            img: Input image (BGR, uint8)
        
        Returns:
            Upscaled image (BGR, uint8)
        """
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            img_tensor = img_tensor.half()
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Convert back
        output = output.squeeze(0).float().cpu().clamp(0, 1).numpy()
        output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def upscale_file(self, input_path: str, output_path: str) -> dict:
        """
        Upscale an image file.
        
        Returns:
            Dict with timing and size info
        """
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")
        
        h, w = img.shape[:2]
        
        start = time.time()
        output = self.upscale(img)
        elapsed_ms = (time.time() - start) * 1000
        
        cv2.imwrite(output_path, output)
        
        out_h, out_w = output.shape[:2]
        
        return {
            "input_size": f"{w}x{h}",
            "output_size": f"{out_w}x{out_h}",
            "time_ms": elapsed_ms
        }


def main():
    parser = argparse.ArgumentParser(description="Upscale images using trained model")
    parser.add_argument("-i", "--input", required=True, help="Input image or folder")
    parser.add_argument("-o", "--output", required=True, help="Output image or folder")
    parser.add_argument("-m", "--model", default=None, help="Model path (optional)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor")
    args = parser.parse_args()
    
    upscaler = TopazUpscaler(model_path=args.model, scale=args.scale)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = upscaler.upscale_file(str(input_path), str(output_path))
        logger.info(f"Upscaled: {input_path.name}")
        logger.info(f"  {result['input_size']} -> {result['output_size']}")
        logger.info(f"  Time: {result['time_ms']:.1f}ms")
        logger.info(f"  Saved: {output_path}")
    
    elif input_path.is_dir():
        # Folder
        output_path.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        logger.info(f"Processing {len(images)} images...")
        
        total_time = 0
        for i, img_path in enumerate(images, 1):
            out_file = output_path / f"{img_path.stem}_upscaled{img_path.suffix}"
            try:
                result = upscaler.upscale_file(str(img_path), str(out_file))
                total_time += result["time_ms"]
                logger.info(f"[{i}/{len(images)}] {img_path.name}: {result['time_ms']:.1f}ms")
            except Exception as e:
                logger.error(f"Failed {img_path.name}: {e}")
        
        avg_time = total_time / len(images) if images else 0
        logger.info(f"Average time: {avg_time:.1f}ms/image")
        logger.info(f"Results saved to: {output_path}")
    
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

