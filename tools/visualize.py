#!/usr/bin/env python3
"""
DepthForge - Output Visualization & Analysis
=============================================
Visualizes disparity maps, compares GPU vs CPU outputs, and computes
error metrics against ground truth.

Usage:
  python3 visualize.py                              # Default files
  python3 visualize.py --gpu disp_gpu.pgm --gt ground_truth.pgm
  python3 visualize.py --all                        # Full analysis

Requires: numpy, matplotlib (pip install matplotlib)
"""

import argparse
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def read_pgm(filename):
    """Read PGM P5 (binary) file."""
    with open(filename, 'rb') as f:
        magic = f.readline().decode().strip()
        assert magic == 'P5', f"Expected P5, got {magic}"
        
        # Skip comments
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    
    return data


def read_ppm(filename):
    """Read PPM P6 (binary) file."""
    with open(filename, 'rb') as f:
        magic = f.readline().decode().strip()
        assert magic == 'P6', f"Expected P6, got {magic}"
        
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(w * h * 3), dtype=np.uint8).reshape(h, w, 3)
    
    return data


def compute_metrics(predicted, ground_truth, max_disp=128):
    """Compute stereo matching quality metrics."""
    # Create valid mask (exclude border regions and zero-disparity)
    valid = (ground_truth > 0) & (ground_truth < max_disp)
    
    if valid.sum() == 0:
        return {}
    
    pred_valid = predicted[valid].astype(np.float64)
    gt_valid = ground_truth[valid].astype(np.float64)
    
    # Absolute error
    abs_err = np.abs(pred_valid - gt_valid)
    
    metrics = {
        'MAE': np.mean(abs_err),
        'RMSE': np.sqrt(np.mean(abs_err**2)),
        'Bad1.0 (%)': 100.0 * np.mean(abs_err > 1.0),   # % pixels with error > 1
        'Bad2.0 (%)': 100.0 * np.mean(abs_err > 2.0),   # % pixels with error > 2
        'Bad3.0 (%)': 100.0 * np.mean(abs_err > 3.0),   # % pixels with error > 3
        'Bad5.0 (%)': 100.0 * np.mean(abs_err > 5.0),   # % pixels with error > 5
        'Valid pixels': int(valid.sum()),
        'Max error': float(abs_err.max()),
        'Median error': float(np.median(abs_err)),
    }
    
    return metrics


def create_comparison_figure(left_img, gpu_disp, cpu_disp, color_map,
                              ground_truth=None, output='analysis.png'):
    """Create a comprehensive comparison figure."""
    if not HAS_MPL:
        print("Cannot create figure: matplotlib not installed")
        return
    
    n_rows = 3 if ground_truth is not None else 2
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, hspace=0.3, wspace=0.2)
    
    # Row 1: Input and outputs
    if left_img is not None:
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(left_img, cmap='gray')
        ax.set_title('Left Input Image', fontsize=12)
        ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(gpu_disp, cmap='jet', vmin=0, vmax=gpu_disp.max())
    ax.set_title('GPU Disparity (SGM, 8-path)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if color_map is not None:
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(color_map)
        ax.set_title('Jet Colormap (CUDA kernel)', fontsize=12)
        ax.axis('off')
    
    # Row 2: CPU comparison and histogram
    if cpu_disp is not None:
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(cpu_disp, cmap='jet', vmin=0, vmax=cpu_disp.max())
        ax.set_title('CPU Disparity (4-path)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Difference map
        ax = fig.add_subplot(gs[1, 1])
        diff = np.abs(gpu_disp.astype(np.float32) - cpu_disp.astype(np.float32))
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
        ax.set_title(f'|GPU - CPU| (Mean: {diff.mean():.2f})', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Disparity histogram
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(gpu_disp.ravel(), bins=64, alpha=0.7, label='GPU', color='steelblue')
    if cpu_disp is not None:
        ax.hist(cpu_disp.ravel(), bins=64, alpha=0.5, label='CPU', color='coral')
    ax.set_xlabel('Disparity')
    ax.set_ylabel('Pixel count')
    ax.set_title('Disparity Distribution')
    ax.legend()
    
    # Row 3: Ground truth comparison (if available)
    if ground_truth is not None and n_rows == 3:
        ax = fig.add_subplot(gs[2, 0])
        im = ax.imshow(ground_truth, cmap='jet', vmin=0, vmax=ground_truth.max())
        ax.set_title('Ground Truth', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Error map
        err = np.abs(gpu_disp.astype(np.float32) - ground_truth.astype(np.float32))
        ax = fig.add_subplot(gs[2, 1])
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=err.max())
        ax.set_title(f'Error Map (MAE: {err.mean():.2f})', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Metrics text
        metrics = compute_metrics(gpu_disp, ground_truth)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        text = "Quality Metrics\n" + "=" * 30 + "\n"
        for k, v in metrics.items():
            if isinstance(v, float):
                text += f"{k}: {v:.3f}\n"
            else:
                text += f"{k}: {v}\n"
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Analysis figure saved: {output}")


def main():
    parser = argparse.ArgumentParser(description='DepthForge Visualization')
    parser.add_argument('--gpu', type=str, default='disparity_gpu.pgm')
    parser.add_argument('--cpu', type=str, default='disparity_cpu.pgm')
    parser.add_argument('--color', type=str, default='depthmap_color.ppm')
    parser.add_argument('--left', type=str, default='data/left.pgm')
    parser.add_argument('--gt', type=str, default='data/ground_truth.pgm')
    parser.add_argument('--output', type=str, default='analysis.png')
    parser.add_argument('--all', action='store_true', help='Full analysis mode')
    args = parser.parse_args()
    
    print("\n  DepthForge - Visualization & Analysis")
    print("  " + "=" * 40)
    
    # Load available files
    gpu_disp = read_pgm(args.gpu) if os.path.exists(args.gpu) else None
    cpu_disp = read_pgm(args.cpu) if os.path.exists(args.cpu) else None
    color_map = read_ppm(args.color) if os.path.exists(args.color) else None
    left_img = read_pgm(args.left) if os.path.exists(args.left) else None
    gt = read_pgm(args.gt) if os.path.exists(args.gt) else None
    
    if gpu_disp is None:
        print("  Error: GPU disparity map not found. Run depthforge first.")
        return
    
    print(f"  GPU disparity: {gpu_disp.shape}, range [{gpu_disp.min()}, {gpu_disp.max()}]")
    
    if cpu_disp is not None:
        print(f"  CPU disparity: {cpu_disp.shape}, range [{cpu_disp.min()}, {cpu_disp.max()}]")
    
    if gt is not None:
        metrics = compute_metrics(gpu_disp, gt)
        print(f"\n  Quality Metrics (GPU vs Ground Truth):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.3f}")
            else:
                print(f"    {k:20s}: {v}")
    
    if HAS_MPL:
        create_comparison_figure(left_img, gpu_disp, cpu_disp, color_map, gt, args.output)
    
    print("\n  Done.\n")


if __name__ == '__main__':
    main()
