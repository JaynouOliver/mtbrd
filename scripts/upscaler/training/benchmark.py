#!/usr/bin/env python3
"""
Full Benchmark: Trained Model vs Topaz Reference
Metrics: SSIM, PSNR, LPIPS

Compares:
  Original (LR) --[Our Model]--> Our Output
  Original (LR) --[Topaz]------> Topaz Output (Ground Truth)
  
  Then: Our Output vs Topaz Output

Usage:
    python benchmark.py --model /path/to/checkpoint.pth --samples 50
"""
import sys
import os
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("WARNING: LPIPS not available. Install with: pip install lpips")


def load_model(model_path, device='cuda'):
    """Load Real-ESRGAN model from checkpoint."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    ckpt = torch.load(model_path, map_location='cpu')
    if 'params_ema' in ckpt:
        model.load_state_dict(ckpt['params_ema'])
    elif 'params' in ckpt:
        model.load_state_dict(ckpt['params'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    if device == 'cuda':
        model = model.half()
    return model


def benchmark(model_path, hr_dir, lr_dir, n_samples=50):
    """Run benchmark comparison."""
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print()
    print("Comparison Flow:")
    print("  Original (LR) --[Our Model]--> Our Output")
    print("  Original (LR) --[Topaz]------> Topaz Output (Ground Truth)")
    print("  Metric: Our Output vs Topaz Output")
    print()
    print(f"Model: {model_path}")
    print(f"Samples: {n_samples}")
    print()

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = load_model(model_path, device)
    print("Model loaded.")

    # Load LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print("LPIPS loaded (AlexNet).")
    print()

    # Get samples
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    hr_files = sorted(hr_dir.glob('*.png'))[:n_samples]

    ssim_vals, psnr_vals, lpips_vals, times = [], [], [], []

    print("Running inference and computing metrics...")
    print("-" * 70)

    for i, hr_path in enumerate(hr_files):
        lr_path = lr_dir / hr_path.name
        if not lr_path.exists():
            continue
        
        # Load images
        lr_img = np.array(Image.open(lr_path).convert('RGB'))  # Original
        hr_ref = np.array(Image.open(hr_path).convert('RGB'))  # Topaz output
        
        # Upscale with our model
        t = torch.from_numpy(lr_img.transpose(2, 0, 1)).float() / 255.0
        t = t.unsqueeze(0).to(device)
        if device == 'cuda':
            t = t.half()
        
        start = time.time()
        with torch.no_grad():
            out = model(t)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        pred = out.squeeze(0).float().cpu().clamp(0, 1).numpy()
        pred = (pred.transpose(1, 2, 0) * 255.0).astype(np.uint8)  # Our output
        
        # Match sizes
        h, w = min(pred.shape[0], hr_ref.shape[0]), min(pred.shape[1], hr_ref.shape[1])
        pred = pred[:h, :w]
        hr_ref = hr_ref[:h, :w]
        
        # SSIM
        ssim_val = calc_ssim(hr_ref, pred, data_range=255, channel_axis=2)
        ssim_vals.append(ssim_val)
        
        # PSNR
        psnr_val = calc_psnr(hr_ref, pred, data_range=255)
        psnr_vals.append(psnr_val)
        
        # LPIPS
        if lpips_fn is not None:
            def to_tensor(im):
                t = torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.0
                t = t.unsqueeze(0) * 2.0 - 1.0  # [-1, 1]
                return t.to(device)
            
            pred_t = to_tensor(pred)
            ref_t = to_tensor(hr_ref)
            
            with torch.no_grad():
                lpips_val = lpips_fn(pred_t, ref_t).item()
            lpips_vals.append(lpips_val)
        
        if (i + 1) % 10 == 0:
            lpips_str = f", LPIPS={np.mean(lpips_vals):.4f}" if lpips_vals else ""
            print(f"  [{i+1:3d}/{n_samples}] SSIM={np.mean(ssim_vals):.4f}, PSNR={np.mean(psnr_vals):.2f}dB{lpips_str}")

    # Results
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS: Our Model vs Topaz")
    print("=" * 70)
    print(f"Samples tested: {len(ssim_vals)}")
    print()

    print("SSIM (Structural Similarity - higher is better):")
    print(f"  Mean:  {np.mean(ssim_vals):.4f}")
    print(f"  Std:   {np.std(ssim_vals):.4f}")
    print(f"  Range: [{np.min(ssim_vals):.4f}, {np.max(ssim_vals):.4f}]")
    print()

    print("PSNR (Peak Signal-to-Noise Ratio - higher is better):")
    print(f"  Mean:  {np.mean(psnr_vals):.2f} dB")
    print(f"  Std:   {np.std(psnr_vals):.2f}")
    print(f"  Range: [{np.min(psnr_vals):.2f}, {np.max(psnr_vals):.2f}]")
    print()

    if lpips_vals:
        print("LPIPS (Learned Perceptual Similarity - lower is better):")
        print(f"  Mean:  {np.mean(lpips_vals):.4f}")
        print(f"  Std:   {np.std(lpips_vals):.4f}")
        print(f"  Range: [{np.min(lpips_vals):.4f}, {np.max(lpips_vals):.4f}]")
        print()

    print(f"Inference Time: {np.mean(times):.1f} ms/image")
    print()

    # Target comparison
    print("=" * 70)
    print("TARGET COMPARISON")
    print("=" * 70)
    print(f"SSIM >= 0.88 (minimum):  {'PASS' if np.mean(ssim_vals) >= 0.88 else 'FAIL'} (current: {np.mean(ssim_vals):.4f})")
    print(f"SSIM >= 0.92 (target):   {'PASS' if np.mean(ssim_vals) >= 0.92 else 'FAIL'}")
    if lpips_vals:
        print(f"LPIPS <= 0.12 (minimum): {'PASS' if np.mean(lpips_vals) <= 0.12 else 'FAIL'} (current: {np.mean(lpips_vals):.4f})")
        print(f"LPIPS <= 0.08 (target):  {'PASS' if np.mean(lpips_vals) <= 0.08 else 'FAIL'}")
    print("=" * 70)

    return {
        'ssim': np.mean(ssim_vals),
        'psnr': np.mean(psnr_vals),
        'lpips': np.mean(lpips_vals) if lpips_vals else None,
        'inference_ms': np.mean(times)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--hr_dir', type=str, default='/home/ubuntu/upscaler/datasets/topaz_train/hr_processed')
    parser.add_argument('--lr_dir', type=str, default='/home/ubuntu/upscaler/datasets/topaz_train/lr_processed')
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    
    benchmark(args.model, args.hr_dir, args.lr_dir, args.samples)

