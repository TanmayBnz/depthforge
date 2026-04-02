#!/usr/bin/env python3
"""
DepthForge - KITTI Stereo 2015 Batch Evaluation
================================================
Converts KITTI PNG stereo pairs to PGM, runs depthforge on each,
reads 16-bit sparse GT disparity, and computes aggregate metrics.

Usage:
  python3 tools/kitti_run.py
  python3 tools/kitti_run.py --max-disp 256 --limit 10 --samples 5
"""

import argparse
import os
import sys
import subprocess
import tempfile
import glob
import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


KITTI_DIR   = "data/kitti/training"
IMAGE_L_DIR = os.path.join(KITTI_DIR, "image_2")
IMAGE_R_DIR = os.path.join(KITTI_DIR, "image_3")
GT_DIR      = os.path.join(KITTI_DIR, "disp_noc_0")
DEPTHFORGE  = "./depthforge"


def write_pgm(path, arr):
    h, w = arr.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(arr.astype(np.uint8).tobytes())


def read_pgm(path):
    with open(path, 'rb') as f:
        magic = f.readline().decode().strip()
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        w, h = map(int, line.split())
        f.readline()  # maxval
        return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)


def read_kitti_gt(path):
    """Read KITTI 16-bit PNG disparity. Returns float array; 0 = invalid."""
    arr = np.array(Image.open(path), dtype=np.float32)
    return arr / 256.0  # KITTI encoding: value / 256 = disparity in pixels


def compute_metrics(pred, gt_float, max_disp):
    """
    pred     : uint8 array, values = disparity in pixels [0, max_disp-1]
    gt_float : float array, 0 = invalid pixel
    """
    valid = gt_float > 0
    if valid.sum() == 0:
        return None

    p = pred[valid].astype(np.float32)
    g = gt_float[valid]

    # Clip GT to max_disp range (disparities > max_disp are unestimable)
    in_range = g < max_disp
    p = p[in_range]
    g = g[in_range]

    if len(p) == 0:
        return None

    err = np.abs(p - g)
    return {
        'MAE':         float(np.mean(err)),
        'RMSE':        float(np.sqrt(np.mean(err**2))),
        'Bad1.0':      float(100.0 * np.mean(err > 1.0)),
        'Bad2.0':      float(100.0 * np.mean(err > 2.0)),
        'Bad3.0':      float(100.0 * np.mean(err > 3.0)),
        'Bad5.0':      float(100.0 * np.mean(err > 5.0)),
        'valid_total': int(valid.sum()),
        'valid_range': int(in_range.sum()),
        'median_err':  float(np.median(err)),
    }


def save_sample_figure(scene_id, left, gpu_disp, gt_float, metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].imshow(left, cmap='gray')
    axes[0].set_title(f'Left image  (scene {scene_id:03d})', fontsize=11)
    axes[0].axis('off')

    vmax = max(gpu_disp.max(), int(gt_float.max()) + 1)
    im = axes[1].imshow(gpu_disp, cmap='jet', vmin=0, vmax=vmax)
    axes[1].set_title('GPU disparity (DepthForge SGM-8)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    gt_plot = np.where(gt_float > 0, gt_float, np.nan)
    im2 = axes[2].imshow(gt_plot, cmap='jet', vmin=0, vmax=vmax)
    axes[2].set_title('GT disparity (KITTI sparse LiDAR)', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if metrics:
        fig.text(0.5, 0.01,
                 f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
                 f"Bad1={metrics['Bad1.0']:.1f}%  Bad3={metrics['Bad3.0']:.1f}%  "
                 f"Valid={metrics['valid_range']} px",
                 ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-disp', type=int, default=256,
                        help='Max disparity passed to depthforge (default 256)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N scenes (default: all 200)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample visualizations to save')
    args = parser.parse_args()

    left_files = sorted(glob.glob(os.path.join(IMAGE_L_DIR, "*_10.png")))
    if not left_files:
        print(f"No KITTI images found in {IMAGE_L_DIR}")
        sys.exit(1)

    if args.limit:
        left_files = left_files[:args.limit]

    n = len(left_files)
    print(f"\n  DepthForge KITTI Batch Evaluation")
    print(f"  {'='*40}")
    print(f"  Scenes     : {n}")
    print(f"  Max disp   : {args.max_disp}")
    print(f"  Binary     : {DEPTHFORGE}\n")

    os.makedirs("data/kitti/results", exist_ok=True)

    all_metrics = []
    gpu_times = []
    sample_interval = max(1, n // args.samples)

    with tempfile.TemporaryDirectory() as tmp:
        l_pgm = os.path.join(tmp, "left.pgm")
        r_pgm = os.path.join(tmp, "right.pgm")

        for i, lpath in enumerate(left_files):
            scene_id = int(os.path.basename(lpath).split('_')[0])
            rpath = os.path.join(IMAGE_R_DIR, os.path.basename(lpath))
            gtpath = os.path.join(GT_DIR, os.path.basename(lpath))

            if not os.path.exists(rpath):
                print(f"  [{i+1}/{n}] scene {scene_id:03d}: missing right image, skipping")
                continue

            # Convert to grayscale PGM
            left_gray = np.array(Image.open(lpath).convert('L'))
            right_gray = np.array(Image.open(rpath).convert('L'))
            write_pgm(l_pgm, left_gray)
            write_pgm(r_pgm, right_gray)

            # Run depthforge
            result = subprocess.run(
                [DEPTHFORGE, l_pgm, r_pgm, str(args.max_disp)],
                capture_output=True, text=True
            )

            # Parse GPU total time from stdout
            gpu_ms = None
            for line in result.stdout.splitlines():
                if "GPU Total:" in line:
                    try:
                        gpu_ms = float(line.split()[2])
                    except Exception:
                        pass

            if gpu_ms:
                gpu_times.append(gpu_ms)

            # Load GPU disparity output
            if not os.path.exists("disparity_gpu.pgm"):
                print(f"  [{i+1}/{n}] scene {scene_id:03d}: depthforge failed")
                continue

            gpu_disp = read_pgm("disparity_gpu.pgm")

            # Load and evaluate against GT
            metrics = None
            gt_float = None
            if os.path.exists(gtpath):
                gt_float = read_kitti_gt(gtpath)
                metrics = compute_metrics(gpu_disp, gt_float, args.max_disp)
                if metrics:
                    all_metrics.append(metrics)

            # Save sample visualizations
            if HAS_MPL and (i % sample_interval == 0):
                out_png = f"data/kitti/results/scene_{scene_id:03d}.png"
                save_sample_figure(scene_id, left_gray, gpu_disp, gt_float if gt_float is not None else np.zeros_like(gpu_disp, dtype=np.float32), metrics, out_png)

            fps = 1000.0 / gpu_ms if gpu_ms else 0
            mae_str = f"MAE={metrics['MAE']:.2f}" if metrics else "no GT"
            print(f"  [{i+1:3d}/{n}] scene {scene_id:03d}  {gpu_ms or 0:6.1f} ms ({fps:4.1f} FPS)  {mae_str}")

    # Aggregate metrics
    if all_metrics:
        print(f"\n  {'='*50}")
        print(f"  KITTI Aggregate Results  ({len(all_metrics)} scenes with GT)")
        print(f"  {'='*50}")
        for key in ['MAE','RMSE','Bad1.0','Bad2.0','Bad3.0','Bad5.0','median_err']:
            vals = [m[key] for m in all_metrics]
            label = key if key != 'median_err' else 'Median err'
            unit = ' %' if 'Bad' in key else ' px'
            print(f"  {label:15s}: {np.mean(vals):.3f}{unit}  (min {np.min(vals):.3f}  max {np.max(vals):.3f})")

    if gpu_times:
        print(f"\n  GPU Timing ({len(gpu_times)} scenes)")
        print(f"  Mean  : {np.mean(gpu_times):.1f} ms  ({1000/np.mean(gpu_times):.1f} FPS)")
        print(f"  Median: {np.median(gpu_times):.1f} ms")
        print(f"  Min   : {np.min(gpu_times):.1f} ms")
        print(f"  Max   : {np.max(gpu_times):.1f} ms")

    # Summary bar chart
    if HAS_MPL and all_metrics:
        bad_keys = ['Bad1.0','Bad2.0','Bad3.0','Bad5.0']
        means = [np.mean([m[k] for m in all_metrics]) for k in bad_keys]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(bad_keys, means, color=['#e74c3c','#e67e22','#f1c40f','#2ecc71'])
        axes[0].set_ylabel('Error rate (%)')
        axes[0].set_title('Bad-pixel rates across 200 KITTI scenes')
        axes[0].set_ylim(0, max(means) * 1.3)
        for j, v in enumerate(means):
            axes[0].text(j, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

        maes = [m['MAE'] for m in all_metrics]
        axes[1].hist(maes, bins=30, color='steelblue', edgecolor='white')
        axes[1].set_xlabel('MAE (pixels)')
        axes[1].set_ylabel('Scene count')
        axes[1].set_title(f'Per-scene MAE distribution  (mean={np.mean(maes):.2f} px)')
        axes[1].axvline(np.mean(maes), color='red', linestyle='--', label=f'mean={np.mean(maes):.2f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("data/kitti/results/kitti_summary.png", dpi=140, bbox_inches='tight')
        print(f"\n  Summary chart : data/kitti/results/kitti_summary.png")
        print(f"  Sample images : data/kitti/results/scene_*.png")

    print("\n  Done.\n")


if __name__ == '__main__':
    main()
