# DepthForge

**Real-Time CUDA-Accelerated Semi-Global Matching for Stereo Depth Estimation with Live Visualization**

> Course Mini-Project — School of Computer Science and Engineering  
> Guide: Dr. Arul Elango

| Name | Roll Number | Section |
|------|-------------|---------|
| Tanmay Singh Bains | 230962180 | AIML-C2 33 |
| Aditya Prashant Naidu | 230962238 | AIML-C2 33 |
| Bhavyy Khurana | 230962102 | AIML-C2 33 |

---

## Overview

DepthForge is a complete, hand-tuned CUDA implementation of the Semi-Global Matching (SGM) stereo depth estimation pipeline. Given a rectified stereo image pair, it produces a dense disparity map rendered as a jet colormap heatmap for real-time depth visualization.

### Pipeline Stages

```
Left Image ──┐
              ├──→ Census Transform ──→ Cost Volume ──→ SGM 8-Path ──→ WTA ──→ Median ──→ Jet Colormap
Right Image ─┘      (CUDA Kernel 1)   (Kernel 2)     Aggregation    (K4)    Filter     (Kernel 6)
                                                       (Kernel 3)             (K5)
```

| Stage | CUDA Kernel | Description |
|-------|-------------|-------------|
| 1. Census Transform | `census_transform_kernel` | 9×7 binary descriptor, one thread per pixel |
| 2. Cost Computation | `compute_cost_kernel` | Hamming distance via `__popcll`, one thread per (x,y,d) |
| 3. SGM Aggregation | `sgm_horizontal_kernel`, `sgm_vertical_kernel`, `sgm_diagonal_kernel` | 8-path cost aggregation with shared memory + parallel min-reduction |
| 4. Winner-Takes-All | `wta_kernel` | Argmin across disparity dimension |
| 5. Median Filter | `median_filter_kernel` | 3×3 sorting-network median |
| 6. Jet Colormap | `jet_colormap_kernel` | Disparity → RGB heatmap |

---

## Requirements

- **NVIDIA GPU** with Compute Capability ≥ 6.0 (Pascal or newer)
- **CUDA Toolkit** 11.0+ (tested with 12.x)
- **Python 3** (for test data generation and visualization)
- **GCC/G++** compatible with your CUDA version

Optional:
- `matplotlib` + `numpy` for visualization (`pip install matplotlib numpy`)
- KITTI stereo dataset for real-world evaluation

---

## Quick Start

### 1. Build

```bash
cd depthforge
make
```

To specify your GPU architecture explicitly:

```bash
# For RTX 3060/3070/3080/3090 (Ampere)
make ARCH=-arch=sm_86

# For RTX 4090 (Ada Lovelace)
make ARCH=-arch=sm_89

# For GTX 1080 (Pascal)
make ARCH=-arch=sm_61
```

### 2. Generate Test Data

```bash
python3 tools/generate_stereo.py --width 640 --height 480 --output data/
```

This creates synthetic stereo images with known ground truth disparity.

### 3. Run

```bash
./depthforge data/left.pgm data/right.pgm 128
```

Or use the convenience target:

```bash
make run
```

### 4. Visualize Results

```bash
python3 tools/visualize.py --gt data/ground_truth.pgm
```

---

## Using Real Images (KITTI Dataset)

1. Download stereo pairs from [KITTI Stereo 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
2. Convert PNG to PGM:
   ```bash
   python3 tools/generate_stereo.py --convert path/to/image_0.png
   ```
3. Run:
   ```bash
   ./depthforge image_0.pgm image_1.pgm 128
   ```

---

## Output Files

| File | Description |
|------|-------------|
| `disparity_gpu.pgm` | Raw GPU disparity map (8-path SGM) |
| `disparity_cpu.pgm` | CPU baseline disparity (4-path SGM) |
| `disparity_normalized.pgm` | Contrast-stretched disparity for viewing |
| `depthmap_color.ppm` | Jet colormap visualization (RGB) |
| `analysis.png` | Comparison figure (if visualization is run) |

---

## Architecture & CUDA Design

### Memory Hierarchy Usage

- **Global Memory**: Cost volume (H×W×D), aggregated costs, images
- **Shared Memory**: Previous scanline costs for SGM aggregation (`s_prev[D]`), parallel reduction buffer (`s_min_buf[D]`)
- **Registers**: Per-thread disparity cost, loop variables

### Thread Organization

| Kernel | Grid | Block | Notes |
|--------|------|-------|-------|
| Census Transform | (W/16, H/16) | (16, 16) | One thread per pixel |
| Cost Computation | (W/16, H/16) | (16, 16) | Loop over D per thread |
| SGM Horizontal | (H,) | (D,) | One block per row, threads = disparities |
| SGM Vertical | (W,) | (D,) | One block per column |
| SGM Diagonal | (W+H-1,) | (D,) | One block per diagonal |
| WTA | (W/16, H/16) | (16, 16) | Argmin over D per pixel |
| Median Filter | (W/16, H/16) | (16, 16) | 3×3 window, sorting network |
| Jet Colormap | (W/16, H/16) | (16, 16) | Direct formula mapping |

### SGM Aggregation Strategy

The 8 directions are processed by 8 sequential kernel launches:

```
Path 1: → (Left to Right)      Path 5: ↘ (TL to BR diagonal)
Path 2: ← (Right to Left)      Path 6: ↗ (BR to TL diagonal)
Path 3: ↓ (Top to Bottom)      Path 7: ↙ (BL to TR diagonal)
Path 4: ↑ (Bottom to Top)      Path 8: ↖ (TR to BL diagonal)
```

Within each kernel, independent scanlines execute in parallel. Threads within a block cooperate on disparity processing with shared memory for neighbor access and parallel min-reduction.

---

## Configuration

Key parameters in `depthforge.cu`:

```c
#define DEFAULT_MAX_DISP  128   // Max disparity (power of 2)
#define CENSUS_RADIUS_X   4     // Census half-width (window = 9)
#define CENSUS_RADIUS_Y   3     // Census half-height (window = 7)
#define SGM_P1            7     // Small disparity penalty
#define SGM_P2            86    // Large disparity penalty
```

Tuning tips:
- **P1**: Controls sensitivity to small depth gradients (slanted surfaces). Lower = smoother.
- **P2**: Controls sensitivity to large depth discontinuities (object edges). Higher = more edge-preserving.
- **MAX_DISP**: Must cover the maximum expected pixel shift. For wider baselines, increase this.

---

## Profiling

```bash
# NVIDIA Visual Profiler
make profile

# Nsight Compute (detailed kernel analysis)
make ncu
```

---

## References

1. H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," IEEE TPAMI, 2008.
2. D. Hernandez-Juarez et al., "Embedded Real-Time Stereo Estimation via Semi-Global Matching on the GPU," 2016.
3. J. Kowalczuk et al., "Real-Time Stereo Matching on CUDA," IEEE TCSVT, 2013.
4. A. Geiger et al., "Efficient Large-Scale Stereo Matching," ACCV 2010.
5. NVIDIA, "CUDA C++ Programming Guide," v12.0, 2023.

---

## Project Structure

```
depthforge/
├── src/
│   └── depthforge.cu         # Complete CUDA implementation (all 6 kernels)
├── tools/
│   ├── generate_stereo.py    # Synthetic test data generator
│   └── visualize.py          # Output visualization & metrics
├── data/                     # Generated/downloaded stereo pairs
├── Makefile                  # Build system
└── README.md                 # This file
```
