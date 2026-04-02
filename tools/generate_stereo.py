#!/usr/bin/env python3
"""
DepthForge - Synthetic Stereo Pair Generator
=============================================
Generates a synthetic left/right stereo image pair with known ground truth
disparity for testing and validating the SGM pipeline.

The scene contains:
  - A textured background (far depth, low disparity)
  - Several geometric shapes at various depths (medium/high disparity)
  - Gaussian noise for realistic texture

Usage:
  python3 generate_stereo.py --width 640 --height 480 --output data/
  python3 generate_stereo.py --width 1280 --height 720 --output data/ --max-disp 128

Outputs:
  data/left.pgm           - Left stereo image
  data/right.pgm          - Right stereo image
  data/ground_truth.pgm   - Ground truth disparity map
"""

import argparse
import os
import struct
import numpy as np


def write_pgm(filename, image):
    """Write a grayscale image as PGM P5 (binary)."""
    h, w = image.shape
    with open(filename, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(image.astype(np.uint8).tobytes())
    print(f"  Saved: {filename} ({w}x{h})")


def create_textured_background(height, width, seed=42):
    """Create a richly textured background using multi-scale noise."""
    rng = np.random.RandomState(seed)
    
    # Multi-scale Perlin-like noise
    bg = np.zeros((height, width), dtype=np.float64)
    
    for scale in [4, 8, 16, 32, 64]:
        h_s = max(2, height // scale)
        w_s = max(2, width // scale)
        noise = rng.randn(h_s, w_s)
        
        # Upsample using bilinear-like interpolation
        y_idx = np.linspace(0, h_s - 1, height)
        x_idx = np.linspace(0, w_s - 1, width)
        
        y0 = np.floor(y_idx).astype(int)
        x0 = np.floor(x_idx).astype(int)
        y1 = np.minimum(y0 + 1, h_s - 1)
        x1 = np.minimum(x0 + 1, w_s - 1)
        
        fy = y_idx - y0
        fx = x_idx - x0
        
        upsampled = (noise[y0][:, x0] * (1 - fx[None, :]) * (1 - fy[:, None]) +
                     noise[y0][:, x1] * fx[None, :] * (1 - fy[:, None]) +
                     noise[y1][:, x0] * (1 - fx[None, :]) * fy[:, None] +
                     noise[y1][:, x1] * fx[None, :] * fy[:, None])
        
        bg += upsampled * (scale / 64.0)
    
    # Normalize to [50, 200] range
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    bg = 50 + 150 * bg
    
    return bg


def add_shapes(image, disparity_map, height, width, max_disp, rng):
    """Add geometric shapes at various depths to the image."""
    shapes = []
    
    # Large rectangle (foreground, high disparity)
    rect_d = int(max_disp * 0.7)
    y1, y2 = height // 3, 2 * height // 3
    x1, x2 = width // 4, width // 2
    image[y1:y2, x1:x2] = rng.randint(80, 180, size=(y2-y1, x2-x1))
    disparity_map[y1:y2, x1:x2] = rect_d
    shapes.append(('rectangle', rect_d))
    
    # Circle (mid-ground)
    circle_d = int(max_disp * 0.45)
    cy, cx = height // 2, 3 * width // 4
    radius = min(height, width) // 6
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    image[mask] = rng.randint(100, 200, size=mask.sum())
    disparity_map[mask] = circle_d
    shapes.append(('circle', circle_d))
    
    # Small square (close foreground, highest disparity)
    sq_d = int(max_disp * 0.9)
    sq_size = min(height, width) // 8
    sy, sx = height // 5, 3 * width // 5
    image[sy:sy+sq_size, sx:sx+sq_size] = rng.randint(140, 220, size=(sq_size, sq_size))
    disparity_map[sy:sy+sq_size, sx:sx+sq_size] = sq_d
    shapes.append(('small_square', sq_d))
    
    # Triangle (mid-depth)
    tri_d = int(max_disp * 0.55)
    tri_cx = width // 6
    tri_cy = 3 * height // 4
    tri_size = min(height, width) // 5
    for row in range(tri_size):
        half_w = int((row / tri_size) * (tri_size // 2))
        y = tri_cy - tri_size + row
        if 0 <= y < height:
            x_lo = max(0, tri_cx - half_w)
            x_hi = min(width, tri_cx + half_w + 1)
            if x_lo < x_hi:
                image[y, x_lo:x_hi] = rng.randint(60, 160, size=(x_hi - x_lo,))
                disparity_map[y, x_lo:x_hi] = tri_d
    shapes.append(('triangle', tri_d))
    
    return shapes


def generate_stereo_pair(width=640, height=480, max_disp=128, seed=42):
    """Generate a synthetic stereo pair with ground truth disparity."""
    rng = np.random.RandomState(seed)
    
    print(f"\n  Generating synthetic stereo pair ({width}x{height})...")
    print(f"  Max disparity: {max_disp}")
    
    # Create textured background
    left = create_textured_background(height, width, seed)
    
    # Initialize disparity map (background = low disparity)
    bg_disparity = int(max_disp * 0.1)
    disparity_gt = np.full((height, width), bg_disparity, dtype=np.float64)
    
    # Add shapes at various depths
    shapes = add_shapes(left, disparity_gt, height, width, max_disp, rng)
    
    # Add fine noise for texture (important for stereo matching!)
    left += rng.randn(height, width) * 5
    left = np.clip(left, 0, 255)
    
    # Generate right image by shifting pixels according to disparity
    right = np.zeros_like(left)
    for y in range(height):
        for x in range(width):
            d = int(disparity_gt[y, x])
            xr = x - d
            if 0 <= xr < width:
                right[y, xr] = left[y, x]
    
    # Fill holes in right image (occluded regions)
    for y in range(height):
        last_valid = 128
        for x in range(width):
            if right[y, x] == 0:
                right[y, x] = last_valid
            else:
                last_valid = right[y, x]
    
    # Add slight noise to right image (simulates camera noise)
    right += rng.randn(height, width) * 2
    right = np.clip(right, 0, 255)
    
    left_u8 = left.astype(np.uint8)
    right_u8 = right.astype(np.uint8)
    disp_u8 = np.clip(disparity_gt, 0, 255).astype(np.uint8)
    
    print(f"  Shapes generated:")
    for name, d in shapes:
        print(f"    {name}: disparity = {d}")
    print(f"  Background disparity: {bg_disparity}")
    
    return left_u8, right_u8, disp_u8


def convert_to_pgm(input_path, output_path):
    """Convert an image file to PGM format using PIL/Pillow."""
    try:
        from PIL import Image
        img = Image.open(input_path).convert('L')
        w, h = img.size
        data = np.array(img)
        write_pgm(output_path, data)
        print(f"  Converted: {input_path} -> {output_path}")
    except ImportError:
        print("  Error: Pillow not installed. Install with: pip install Pillow")
        print("  Or use images already in PGM format.")


def main():
    parser = argparse.ArgumentParser(description='DepthForge Stereo Pair Generator')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--max-disp', type=int, default=128, help='Max disparity')
    parser.add_argument('--output', type=str, default='data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--convert', type=str, default=None,
                        help='Convert an image to PGM format')
    args = parser.parse_args()
    
    if args.convert:
        out = args.convert.rsplit('.', 1)[0] + '.pgm'
        convert_to_pgm(args.convert, out)
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    left, right, gt = generate_stereo_pair(
        args.width, args.height, args.max_disp, args.seed
    )
    
    write_pgm(os.path.join(args.output, 'left.pgm'), left)
    write_pgm(os.path.join(args.output, 'right.pgm'), right)
    write_pgm(os.path.join(args.output, 'ground_truth.pgm'), gt)
    
    print(f"\n  Files saved to {args.output}")
    print(f"  Run: ./depthforge {args.output}/left.pgm {args.output}/right.pgm {args.max_disp}")


if __name__ == '__main__':
    main()
