#!/usr/bin/env python3"""
DepthForge vs OpenCV StereoSGBM Baseline
=========================================
Runs OpenCV StereoBM and StereoSGBM on all 200 KITTI Stereo 2015 training
scenes and computes the same metrics as kitti_run.py for direct comparison.
"""

import glob, os, time
import numpy as np
import cv2
from PIL import Image

KITTI_DIR   = "data/kitti/training"
IMAGE_L_DIR = os.path.join(KITTI_DIR, "image_2")
IMAGE_R_DIR = os.path.join(KITTI_DIR, "image_3")
GT_DIR      = os.path.join(KITTI_DIR, "disp_noc_0")
MAX_DISP    = 256


def read_kitti_gt(path):
    arr = np.array(Image.open(path), dtype=np.float32)
    return arr / 256.0


def compute_metrics(pred_float, gt_float):
    valid = gt_float > 0
    in_range = valid & (gt_float < MAX_DISP)
    if in_range.sum() == 0:
        return None
    p = pred_float[in_range]
    g = gt_float[in_range]
    err = np.abs(p - g)
    return {
        'MAE':      float(np.mean(err)),
        'RMSE':     float(np.sqrt(np.mean(err**2))),
        'Bad1.0':   float(100.0 * np.mean(err > 1.0)),
        'Bad2.0':   float(100.0 * np.mean(err > 2.0)),
        'Bad3.0':   float(100.0 * np.mean(err > 3.0)),
        'Bad5.0':   float(100.0 * np.mean(err > 5.0)),
        'median':   float(np.median(err)),
    }


def run_sgbm(left_gray, right_gray):
    """OpenCV StereoSGBM — tuned to match DepthForge parameters as closely as possible."""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=MAX_DISP,
        blockSize=9,            # similar to DepthForge 9x7 census window
        P1=7 * 9 * 9,           # P1 scaled by blockSize^2 (OpenCV convention)
        P2=86 * 9 * 9,          # P2 scaled similarly
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # 3-direction variant (5-path internally)
    )
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def run_bm(left_gray, right_gray):
    """OpenCV StereoBM — fast block matching baseline."""
    bm = cv2.StereoBM_create(numDisparities=MAX_DISP, blockSize=21)
    disp = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def aggregate(metrics_list, label):
    if not metrics_list:
        return
    print(f"\n  {label}  ({len(metrics_list)} scenes)")
    print(f"  {'-'*50}")
    for key in ['MAE','RMSE','Bad1.0','Bad2.0','Bad3.0','Bad5.0','median']:
        vals = [m[key] for m in metrics_list]
        unit = ' %' if 'Bad' in key else ' px'
        print(f"  {key:12s}: {np.mean(vals):.3f}{unit}  (min {np.min(vals):.3f}  max {np.max(vals):.3f})")


def main():
    left_files = sorted(glob.glob(os.path.join(IMAGE_L_DIR, "*_10.png")))
    n = len(left_files)
    print(f"\n  OpenCV Stereo Baseline — KITTI ({n} scenes, max_disp={MAX_DISP})\n")

    sgbm_metrics, bm_metrics = [], []
    sgbm_times, bm_times = [], []

    for i, lpath in enumerate(left_files):
        rpath = os.path.join(IMAGE_R_DIR, os.path.basename(lpath))
        gtpath = os.path.join(GT_DIR, os.path.basename(lpath))
        if not os.path.exists(rpath):
            continue

        left  = np.array(Image.open(lpath).convert('L'))
        right = np.array(Image.open(rpath).convert('L'))

        # StereoBM
        t0 = time.perf_counter()
        bm_disp = run_bm(left, right)
        bm_ms = (time.perf_counter() - t0) * 1000
        bm_times.append(bm_ms)

        # StereoSGBM
        t0 = time.perf_counter()
        sgbm_disp = run_sgbm(left, right)
        sgbm_ms = (time.perf_counter() - t0) * 1000
        sgbm_times.append(sgbm_ms)

        if os.path.exists(gtpath):
            gt = read_kitti_gt(gtpath)
            m_bm   = compute_metrics(bm_disp, gt)
            m_sgbm = compute_metrics(sgbm_disp, gt)
            if m_bm:   bm_metrics.append(m_bm)
            if m_sgbm: sgbm_metrics.append(m_sgbm)

        print(f"  [{i+1:3d}/{n}]  BM {bm_ms:6.1f}ms MAE={bm_metrics[-1]['MAE']:.2f}   "
              f"SGBM {sgbm_ms:7.1f}ms MAE={sgbm_metrics[-1]['MAE']:.2f}")

    # Timing summary
    print(f"\n  {'='*55}")
    print(f"  Timing Summary")
    print(f"  {'='*55}")
    print(f"  StereoBM   mean {np.mean(bm_times):.0f}ms  median {np.median(bm_times):.0f}ms")
    print(f"  StereoSGBM mean {np.mean(sgbm_times):.0f}ms  median {np.median(sgbm_times):.0f}ms")

    aggregate(bm_metrics,   "OpenCV StereoBM")
    aggregate(sgbm_metrics, "OpenCV StereoSGBM")

    print()


if __name__ == '__main__':
    main()
