/*
 * ============================================================================
 * DepthForge - Real-Time CUDA-Accelerated Semi-Global Matching
 * for Stereo Depth Estimation with Jet Colormap Visualization
 * ============================================================================
 *
 * Authors:  Tanmay Singh Bains    (230962180, AIML-C2 33)
 *           Aditya Prashant Naidu (230962238, AIML-C2 33)
 *           Bhavyy Khurana        (230962102, AIML-C2 33)
 *
 * Guide:    Dr. Arul Elango
 * School:   School of Computer Science and Engineering
 *
 * Description:
 *   End-to-end CUDA implementation of the Semi-Global Matching (SGM) stereo
 *   depth estimation pipeline. The pipeline consists of:
 *     1. Census Transform - robust illumination-invariant cost computation
 *     2. Matching Cost Volume - Hamming distance between census descriptors
 *     3. SGM Cost Aggregation - 8-path directional cost smoothing
 *     4. Winner-Takes-All    - disparity selection via parallel reduction
 *     5. Median Filter       - post-processing noise removal
 *     6. Jet Colormap        - disparity-to-color visualization
 *
 *   Includes CPU baseline for benchmarking comparison.
 *
 * References:
 *   [1] Hirschmuller, 2008 - Semi-Global Matching
 *   [2] Hernandez-Juarez et al., 2016 - Embedded SGM on CUDA
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -o depthforge src/depthforge.cu
 *
 * Usage:
 *   ./depthforge <left.pgm> <right.pgm> [max_disparity]
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================================
// Configuration
// ============================================================================

#define DEFAULT_MAX_DISP  128   // Maximum disparity levels
#define CENSUS_RADIUS_X   4     // Census window half-width  (full = 9)
#define CENSUS_RADIUS_Y   3     // Census window half-height (full = 7)
#define CENSUS_WIDTH       (2 * CENSUS_RADIUS_X + 1)  // 9
#define CENSUS_HEIGHT      (2 * CENSUS_RADIUS_Y + 1)  // 7
#define CENSUS_BITS        (CENSUS_WIDTH * CENSUS_HEIGHT - 1)  // 62 bits

#define SGM_P1            7     // Small penalty for +-1 disparity change
#define SGM_P2            86    // Large penalty for >1 disparity change
#define NUM_PATHS         8     // Number of SGM aggregation directions

#define BLOCK_SIZE        128   // CUDA block size for 1D kernels
#define BLOCK_2D          16    // CUDA block size for 2D kernels

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer utility
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() { cudaEventRecord(start, 0); }
    void Stop()  { cudaEventRecord(stop, 0); cudaEventSynchronize(stop); }
    float ElapsedMs() {
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Image I/O (PGM / PPM format)
// ============================================================================

// Read PGM (P5 binary grayscale) image
uint8_t* read_pgm(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: %s is not a PGM (P5) file\n", filename);
        fclose(f);
        return NULL;
    }

    // Skip comments
    int c = fgetc(f);
    while (c == '#' || c == '\n' || c == '\r' || c == ' ') {
        if (c == '#') while (fgetc(f) != '\n');
        c = fgetc(f);
    }
    ungetc(c, f);

    int maxval;
    if (fscanf(f, "%d %d %d", width, height, &maxval) != 3) {
        fprintf(stderr, "Error: Invalid PGM header in %s\n", filename);
        fclose(f);
        return NULL;
    }
    fgetc(f); // consume single whitespace after maxval

    int size = (*width) * (*height);
    uint8_t* data = (uint8_t*)malloc(size);
    if (fread(data, 1, size, f) != (size_t)size) {
        fprintf(stderr, "Error: Failed to read pixel data from %s\n", filename);
        free(data);
        fclose(f);
        return NULL;
    }

    fclose(f);
    printf("  Loaded: %s (%dx%d)\n", filename, *width, *height);
    return data;
}

// Write PGM (P5 binary grayscale) image
void write_pgm(const char* filename, const uint8_t* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, f);
    fclose(f);
    printf("  Saved: %s (%dx%d)\n", filename, width, height);
}

// Write PPM (P6 binary RGB) image
void write_ppm(const char* filename, const uint8_t* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height * 3, f);
    fclose(f);
    printf("  Saved: %s (%dx%d, RGB)\n", filename, width, height);
}


// ============================================================================
// KERNEL 1: Census Transform
// ============================================================================
//
// Each thread computes a 62-bit census descriptor for one pixel.
// The descriptor encodes the local intensity structure by comparing
// each neighbor pixel against the center pixel. This provides
// robustness to illumination changes (additive/multiplicative).
//
// Window: 9x7 (62 comparisons, stored in uint64_t)
// Thread mapping: one thread per pixel
// ============================================================================

__global__ void census_transform_kernel(
    const uint8_t* __restrict__ image,
    uint64_t*      __restrict__ census,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t center = image[y * width + x];
    uint64_t descriptor = 0;
    int bit = 0;

    for (int dy = -CENSUS_RADIUS_Y; dy <= CENSUS_RADIUS_Y; dy++) {
        for (int dx = -CENSUS_RADIUS_X; dx <= CENSUS_RADIUS_X; dx++) {
            if (dx == 0 && dy == 0) continue; // skip center

            int nx = x + dx;
            int ny = y + dy;

            // Clamp to image bounds
            nx = max(0, min(width - 1, nx));
            ny = max(0, min(height - 1, ny));

            uint8_t neighbor = image[ny * width + nx];

            // Set bit if neighbor < center
            if (neighbor < center) {
                descriptor |= (1ULL << bit);
            }
            bit++;
        }
    }

    census[y * width + x] = descriptor;
}


// ============================================================================
// KERNEL 2: Matching Cost Computation (Hamming Distance)
// ============================================================================
//
// For each pixel (x, y) and disparity d, compute the Hamming distance
// between the left census descriptor at (x, y) and the right census
// descriptor at (x - d, y). Hamming distance = popcount(XOR).
//
// Output: 3D cost volume of shape [H x W x D], stored as uint8_t
//         (max Hamming distance = 62, fits in uint8_t).
// Thread mapping: one thread per (x, y, d) triplet.
// ============================================================================

__global__ void compute_cost_kernel(
    const uint64_t* __restrict__ census_left,
    const uint64_t* __restrict__ census_right,
    uint8_t*        __restrict__ cost_volume,
    int width, int height, int max_disp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint64_t left_desc = census_left[y * width + x];

    for (int d = 0; d < max_disp; d++) {
        int xr = x - d;
        uint8_t cost;

        if (xr < 0) {
            cost = CENSUS_BITS; // Maximum cost for out-of-bounds
        } else {
            uint64_t right_desc = census_right[y * width + xr];
            cost = (uint8_t)__popcll(left_desc ^ right_desc);
        }

        cost_volume[(y * width + x) * max_disp + d] = cost;
    }
}


// ============================================================================
// KERNEL 3: SGM Cost Aggregation
// ============================================================================
//
// The core of SGM: aggregate matching costs along 8 directional paths.
// For each path direction r, the aggregated cost at pixel p for disparity d:
//
//   Lr(p, d) = C(p, d) + min(
//       Lr(p-r, d),           // same disparity
//       Lr(p-r, d-1) + P1,    // +-1 disparity
//       Lr(p-r, d+1) + P1,
//       min_k Lr(p-r, k) + P2 // all other disparities
//   ) - min_k Lr(p-r, k)
//
// The subtraction of min_k prevents path costs from growing unbounded.
//
// Implementation: One kernel per direction. For horizontal paths, one block
// per row with threads handling disparities in parallel. For vertical and
// diagonal paths, similar scanline parallelism.
//
// Shared memory is used to store the previous pixel's path costs for
// efficient neighbor access across disparities.
// ============================================================================

// Horizontal aggregation: left-to-right (dir=1) or right-to-left (dir=-1)
__global__ void sgm_horizontal_kernel(
    const uint8_t*  __restrict__ cost_volume,
    uint32_t*       __restrict__ agg_cost,
    int width, int height, int max_disp,
    int direction  // +1 = left-to-right, -1 = right-to-left
) {
    // One block per row, threads cover disparities
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= max_disp) return;

    extern __shared__ uint16_t smem[];
    uint16_t* s_prev = smem;                    // [max_disp]
    uint16_t* s_min_buf = smem + max_disp;      // [max_disp] for reduction

    int x_start = (direction == 1) ? 0 : width - 1;
    int x_end   = (direction == 1) ? width : -1;

    // Initialize first pixel on this scanline
    int idx0 = (y * width + x_start) * max_disp + d;
    uint16_t init_cost = (uint16_t)cost_volume[idx0];
    s_prev[d] = init_cost;
    atomicAdd(&agg_cost[idx0], init_cost);
    __syncthreads();

    // Compute min of previous costs via parallel reduction
    s_min_buf[d] = s_prev[d];
    __syncthreads();
    for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
        if (d < stride && (d + stride) < max_disp) {
            s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
        }
        __syncthreads();
    }
    uint16_t min_prev = s_min_buf[0];
    __syncthreads();

    // Process remaining pixels along the scanline
    for (int x = x_start + direction; x != x_end; x += direction) {
        int idx = (y * width + x) * max_disp + d;
        uint16_t c = (uint16_t)cost_volume[idx];

        // SGM recurrence
        uint16_t l0 = s_prev[d];                                           // same disparity
        uint16_t l1 = (d > 0)          ? s_prev[d - 1] + SGM_P1 : 0xFFFF; // d-1
        uint16_t l2 = (d < max_disp-1) ? s_prev[d + 1] + SGM_P1 : 0xFFFF; // d+1
        uint16_t l3 = min_prev + SGM_P2;                                   // all others

        uint16_t path_cost = c + min(min(l0, l1), min(l2, l3)) - min_prev;

        // Accumulate into aggregated cost volume
        atomicAdd(&agg_cost[idx], path_cost);

        // Update shared memory for next iteration
        s_prev[d] = path_cost;
        __syncthreads();

        // Parallel min reduction
        s_min_buf[d] = path_cost;
        __syncthreads();
        for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
            if (d < stride && (d + stride) < max_disp) {
                s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
            }
            __syncthreads();
        }
        min_prev = s_min_buf[0];
        __syncthreads();
    }
}

// Vertical aggregation: top-to-bottom (dir=1) or bottom-to-top (dir=-1)
__global__ void sgm_vertical_kernel(
    const uint8_t*  __restrict__ cost_volume,
    uint32_t*       __restrict__ agg_cost,
    int width, int height, int max_disp,
    int direction  // +1 = top-to-bottom, -1 = bottom-to-top
) {
    // One block per column, threads cover disparities
    int x = blockIdx.x;
    int d = threadIdx.x;
    if (x >= width || d >= max_disp) return;

    extern __shared__ uint16_t smem[];
    uint16_t* s_prev = smem;
    uint16_t* s_min_buf = smem + max_disp;

    int y_start = (direction == 1) ? 0 : height - 1;
    int y_end   = (direction == 1) ? height : -1;

    // Initialize
    int idx0 = (y_start * width + x) * max_disp + d;
    uint16_t init_cost = (uint16_t)cost_volume[idx0];
    s_prev[d] = init_cost;
    atomicAdd(&agg_cost[idx0], init_cost);
    __syncthreads();

    s_min_buf[d] = s_prev[d];
    __syncthreads();
    for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
        if (d < stride && (d + stride) < max_disp)
            s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
        __syncthreads();
    }
    uint16_t min_prev = s_min_buf[0];
    __syncthreads();

    for (int y = y_start + direction; y != y_end; y += direction) {
        int idx = (y * width + x) * max_disp + d;
        uint16_t c = (uint16_t)cost_volume[idx];

        uint16_t l0 = s_prev[d];
        uint16_t l1 = (d > 0)          ? s_prev[d - 1] + SGM_P1 : 0xFFFF;
        uint16_t l2 = (d < max_disp-1) ? s_prev[d + 1] + SGM_P1 : 0xFFFF;
        uint16_t l3 = min_prev + SGM_P2;

        uint16_t path_cost = c + min(min(l0, l1), min(l2, l3)) - min_prev;
        atomicAdd(&agg_cost[idx], path_cost);

        s_prev[d] = path_cost;
        __syncthreads();

        s_min_buf[d] = path_cost;
        __syncthreads();
        for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
            if (d < stride && (d + stride) < max_disp)
                s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
            __syncthreads();
        }
        min_prev = s_min_buf[0];
        __syncthreads();
    }
}

// Diagonal aggregation: handles all 4 diagonal directions
// dir_x, dir_y define the direction: (+1,+1), (-1,-1), (+1,-1), (-1,+1)
__global__ void sgm_diagonal_kernel(
    const uint8_t*  __restrict__ cost_volume,
    uint32_t*       __restrict__ agg_cost,
    int width, int height, int max_disp,
    int dir_x, int dir_y
) {
    // Each block handles one diagonal scanline
    // Diagonals are indexed along the starting edge
    int diag_id = blockIdx.x;
    int d = threadIdx.x;
    if (d >= max_disp) return;

    extern __shared__ uint16_t smem[];
    uint16_t* s_prev = smem;
    uint16_t* s_min_buf = smem + max_disp;

    // Determine starting pixel for this diagonal
    int x_start, y_start;
    int total_diags = width + height - 1;
    if (diag_id >= total_diags) return;

    if (dir_x == 1 && dir_y == 1) {
        // Top-left to bottom-right: start from top row and left column
        if (diag_id < width) {
            x_start = diag_id; y_start = 0;
        } else {
            x_start = 0; y_start = diag_id - width + 1;
        }
    } else if (dir_x == -1 && dir_y == -1) {
        // Bottom-right to top-left: start from bottom row and right column
        if (diag_id < width) {
            x_start = width - 1 - diag_id; y_start = height - 1;
        } else {
            x_start = width - 1; y_start = height - 1 - (diag_id - width + 1);
        }
    } else if (dir_x == 1 && dir_y == -1) {
        // Bottom-left to top-right: start from bottom row and left column
        if (diag_id < width) {
            x_start = diag_id; y_start = height - 1;
        } else {
            x_start = 0; y_start = height - 1 - (diag_id - width + 1);
        }
    } else { // dir_x == -1, dir_y == 1
        // Top-right to bottom-left: start from top row and right column
        if (diag_id < width) {
            x_start = width - 1 - diag_id; y_start = 0;
        } else {
            x_start = width - 1; y_start = diag_id - width + 1;
        }
    }

    // Check bounds
    if (x_start < 0 || x_start >= width || y_start < 0 || y_start >= height) return;

    // Initialize first pixel
    int idx0 = (y_start * width + x_start) * max_disp + d;
    uint16_t init_cost = (uint16_t)cost_volume[idx0];
    s_prev[d] = init_cost;
    atomicAdd(&agg_cost[idx0], init_cost);
    __syncthreads();

    s_min_buf[d] = s_prev[d];
    __syncthreads();
    for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
        if (d < stride && (d + stride) < max_disp)
            s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
        __syncthreads();
    }
    uint16_t min_prev = s_min_buf[0];
    __syncthreads();

    // Walk along diagonal
    int x = x_start + dir_x;
    int y = y_start + dir_y;

    while (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = (y * width + x) * max_disp + d;
        uint16_t c = (uint16_t)cost_volume[idx];

        uint16_t l0 = s_prev[d];
        uint16_t l1 = (d > 0)          ? s_prev[d - 1] + SGM_P1 : 0xFFFF;
        uint16_t l2 = (d < max_disp-1) ? s_prev[d + 1] + SGM_P1 : 0xFFFF;
        uint16_t l3 = min_prev + SGM_P2;

        uint16_t path_cost = c + min(min(l0, l1), min(l2, l3)) - min_prev;
        atomicAdd(&agg_cost[idx], path_cost);

        s_prev[d] = path_cost;
        __syncthreads();

        s_min_buf[d] = path_cost;
        __syncthreads();
        for (int stride = max_disp / 2; stride > 0; stride >>= 1) {
            if (d < stride && (d + stride) < max_disp)
                s_min_buf[d] = min(s_min_buf[d], s_min_buf[d + stride]);
            __syncthreads();
        }
        min_prev = s_min_buf[0];
        __syncthreads();

        x += dir_x;
        y += dir_y;
    }
}


// ============================================================================
// KERNEL 4: Winner-Takes-All Disparity Selection
// ============================================================================
//
// For each pixel, find the disparity with minimum aggregated cost.
// Uses a simple argmin across the disparity dimension.
// ============================================================================

__global__ void wta_kernel(
    const uint32_t* __restrict__ agg_cost,
    uint8_t*        __restrict__ disparity_map,
    int width, int height, int max_disp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int base = (y * width + x) * max_disp;
    uint32_t min_cost = agg_cost[base];
    uint8_t best_d = 0;

    for (int d = 1; d < max_disp; d++) {
        uint32_t cost = agg_cost[base + d];
        if (cost < min_cost) {
            min_cost = cost;
            best_d = (uint8_t)d;
        }
    }

    disparity_map[y * width + x] = best_d;
}


// ============================================================================
// KERNEL 5: Median Filter (3x3)
// ============================================================================
//
// Post-processing step to remove salt-and-pepper noise from the
// disparity map. Uses a simple 3x3 window with sorting network.
// ============================================================================

__device__ void sort2(uint8_t& a, uint8_t& b) {
    if (a > b) { uint8_t t = a; a = b; b = t; }
}

__global__ void median_filter_kernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Gather 3x3 neighborhood
    uint8_t window[9];
    int k = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));
            window[k++] = input[ny * width + nx];
        }
    }

    // Partial sorting network to find median (5th element of 9)
    sort2(window[0], window[1]); sort2(window[3], window[4]);
    sort2(window[6], window[7]); sort2(window[1], window[2]);
    sort2(window[4], window[5]); sort2(window[7], window[8]);
    sort2(window[0], window[1]); sort2(window[3], window[4]);
    sort2(window[6], window[7]); sort2(window[0], window[3]);
    sort2(window[3], window[6]); sort2(window[1], window[4]);
    sort2(window[4], window[7]); sort2(window[2], window[5]);
    sort2(window[5], window[8]); sort2(window[1], window[3]);
    sort2(window[5], window[7]); sort2(window[2], window[6]);
    sort2(window[4], window[6]); sort2(window[2], window[4]);
    sort2(window[2], window[3]); sort2(window[4], window[5]);

    output[y * width + x] = window[4]; // median
}


// ============================================================================
// KERNEL 6: Jet Colormap Visualization
// ============================================================================
//
// Maps disparity values to RGB using the jet colormap for intuitive
// depth visualization. Red = close (high disparity), blue = far (low).
// ============================================================================

__device__ void jet_colormap(float value, uint8_t& r, uint8_t& g, uint8_t& b) {
    // value in [0, 1]
    float v = fminf(fmaxf(value, 0.0f), 1.0f);

    float r_f = 0.0f, g_f = 0.0f, b_f = 0.0f;

    if (v < 0.125f) {
        r_f = 0.0f;
        g_f = 0.0f;
        b_f = 0.5f + 0.5f * (v / 0.125f);
    } else if (v < 0.375f) {
        r_f = 0.0f;
        g_f = (v - 0.125f) / 0.25f;
        b_f = 1.0f;
    } else if (v < 0.625f) {
        r_f = (v - 0.375f) / 0.25f;
        g_f = 1.0f;
        b_f = 1.0f - (v - 0.375f) / 0.25f;
    } else if (v < 0.875f) {
        r_f = 1.0f;
        g_f = 1.0f - (v - 0.625f) / 0.25f;
        b_f = 0.0f;
    } else {
        r_f = 1.0f - 0.5f * (v - 0.875f) / 0.125f;
        g_f = 0.0f;
        b_f = 0.0f;
    }

    r = (uint8_t)(r_f * 255.0f);
    g = (uint8_t)(g_f * 255.0f);
    b = (uint8_t)(b_f * 255.0f);
}

__global__ void jet_colormap_kernel(
    const uint8_t* __restrict__ disparity_map,
    uint8_t*       __restrict__ color_output,   // RGB interleaved
    int width, int height, int max_disp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t disp = disparity_map[y * width + x];
    float normalized = (float)disp / (float)(max_disp - 1);

    uint8_t r, g, b;
    jet_colormap(normalized, r, g, b);

    int out_idx = (y * width + x) * 3;
    color_output[out_idx + 0] = r;
    color_output[out_idx + 1] = g;
    color_output[out_idx + 2] = b;
}


// ============================================================================
// CPU Baseline Implementation
// ============================================================================

void cpu_census_transform(const uint8_t* image, uint64_t* census, int W, int H) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t center = image[y * W + x];
            uint64_t desc = 0;
            int bit = 0;
            for (int dy = -CENSUS_RADIUS_Y; dy <= CENSUS_RADIUS_Y; dy++) {
                for (int dx = -CENSUS_RADIUS_X; dx <= CENSUS_RADIUS_X; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = max(0, min(W - 1, x + dx));
                    int ny = max(0, min(H - 1, y + dy));
                    if (image[ny * W + nx] < center)
                        desc |= (1ULL << bit);
                    bit++;
                }
            }
            census[y * W + x] = desc;
        }
    }
}

int popcount64(uint64_t x) {
    int count = 0;
    while (x) { count += (x & 1); x >>= 1; }
    return count;
}

void cpu_compute_cost(const uint64_t* cl, const uint64_t* cr,
                      uint8_t* cost, int W, int H, int D) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int d = 0; d < D; d++) {
                int xr = x - d;
                if (xr < 0)
                    cost[(y * W + x) * D + d] = CENSUS_BITS;
                else
                    cost[(y * W + x) * D + d] =
                        (uint8_t)popcount64(cl[y * W + x] ^ cr[y * W + xr]);
            }
}

void cpu_sgm_aggregate(const uint8_t* cost, uint16_t* agg,
                       int W, int H, int D) {
    // Simplified: only 4 paths (L-R, R-L, T-B, B-T) for CPU baseline
    memset(agg, 0, (size_t)W * H * D * sizeof(uint16_t));

    // Temporary buffer for one path
    uint16_t* prev = (uint16_t*)malloc(D * sizeof(uint16_t));

    // Direction vectors: dx, dy
    int dirs[][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

    for (int dir = 0; dir < 4; dir++) {
        int dx = dirs[dir][0], dy = dirs[dir][1];

        for (int start = 0; start < (dx != 0 ? H : W); start++) {
            int x = (dx ==  1) ? 0 : (dx == -1) ? W-1 : start;
            int y = (dy ==  1) ? 0 : (dy == -1) ? H-1 : start;

            // Initialize
            for (int d = 0; d < D; d++) {
                prev[d] = cost[(y * W + x) * D + d];
                agg[(y * W + x) * D + d] += prev[d];
            }

            uint16_t min_prev = prev[0];
            for (int d = 1; d < D; d++)
                if (prev[d] < min_prev) min_prev = prev[d];

            x += dx; y += dy;

            while (x >= 0 && x < W && y >= 0 && y < H) {
                uint16_t min_cur = 0xFFFF;
                uint16_t cur[256]; // max disparity

                for (int d = 0; d < D; d++) {
                    uint16_t c = cost[(y * W + x) * D + d];
                    uint16_t l0 = prev[d];
                    uint16_t l1 = (d > 0)   ? prev[d-1] + SGM_P1 : 0xFFFF;
                    uint16_t l2 = (d < D-1) ? prev[d+1] + SGM_P1 : 0xFFFF;
                    uint16_t l3 = min_prev + SGM_P2;

                    cur[d] = c + min(min(l0, l1), min(l2, l3)) - min_prev;
                    agg[(y * W + x) * D + d] += cur[d];
                    if (cur[d] < min_cur) min_cur = cur[d];
                }

                memcpy(prev, cur, D * sizeof(uint16_t));
                min_prev = min_cur;
                x += dx; y += dy;
            }
        }
    }
    free(prev);
}

void cpu_wta(const uint16_t* agg, uint8_t* disp, int W, int H, int D) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int base = (y * W + x) * D;
            uint16_t best = agg[base];
            uint8_t best_d = 0;
            for (int d = 1; d < D; d++) {
                if (agg[base + d] < best) {
                    best = agg[base + d];
                    best_d = d;
                }
            }
            disp[y * W + x] = best_d;
        }
}


// ============================================================================
// GPU Pipeline Orchestration
// ============================================================================

void run_gpu_pipeline(
    const uint8_t* h_left, const uint8_t* h_right,
    uint8_t* h_disparity, uint8_t* h_color,
    int width, int height, int max_disp,
    float* stage_times  // Array of 6 floats for per-stage timing
) {
    size_t img_size     = (size_t)width * height;
    size_t census_size  = img_size * sizeof(uint64_t);
    size_t cost_size    = img_size * max_disp * sizeof(uint8_t);
    size_t agg_size     = img_size * max_disp * sizeof(uint32_t);
    size_t color_size   = img_size * 3;

    printf("\n  Memory allocation:\n");
    printf("    Image:        %zu bytes\n", img_size);
    printf("    Census:       %zu bytes (x2)\n", census_size);
    printf("    Cost volume:  %zu bytes (%.1f MB)\n", cost_size, cost_size/1e6);
    printf("    Agg. volume:  %zu bytes (%.1f MB)\n", agg_size, agg_size/1e6);

    // Device memory allocation
    uint8_t  *d_left, *d_right;
    uint64_t *d_census_left, *d_census_right;
    uint8_t  *d_cost_volume;
    uint32_t *d_agg_cost;
    uint8_t  *d_disparity, *d_disparity_filtered;
    uint8_t  *d_color;

    CUDA_CHECK(cudaMalloc(&d_left,              img_size));
    CUDA_CHECK(cudaMalloc(&d_right,             img_size));
    CUDA_CHECK(cudaMalloc(&d_census_left,       census_size));
    CUDA_CHECK(cudaMalloc(&d_census_right,      census_size));
    CUDA_CHECK(cudaMalloc(&d_cost_volume,       cost_size));
    CUDA_CHECK(cudaMalloc(&d_agg_cost,          agg_size));
    CUDA_CHECK(cudaMalloc(&d_disparity,         img_size));
    CUDA_CHECK(cudaMalloc(&d_disparity_filtered,img_size));
    CUDA_CHECK(cudaMalloc(&d_color,             color_size));

    // Copy input images to device
    CUDA_CHECK(cudaMemcpy(d_left,  h_left,  img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, h_right, img_size, cudaMemcpyHostToDevice));

    // Initialize aggregated cost to zero
    CUDA_CHECK(cudaMemset(d_agg_cost, 0, agg_size));

    GpuTimer timer;
    dim3 block2d(BLOCK_2D, BLOCK_2D);
    dim3 grid2d((width + BLOCK_2D - 1) / BLOCK_2D,
                (height + BLOCK_2D - 1) / BLOCK_2D);

    // ---- Stage 1: Census Transform ----
    printf("\n  [Stage 1] Census Transform...\n");
    timer.Start();
    census_transform_kernel<<<grid2d, block2d>>>(d_left,  d_census_left,  width, height);
    census_transform_kernel<<<grid2d, block2d>>>(d_right, d_census_right, width, height);
    timer.Stop();
    stage_times[0] = timer.ElapsedMs();
    printf("    Time: %.3f ms\n", stage_times[0]);

    // ---- Stage 2: Cost Computation ----
    printf("  [Stage 2] Cost Computation (Hamming distance)...\n");
    timer.Start();
    compute_cost_kernel<<<grid2d, block2d>>>(
        d_census_left, d_census_right, d_cost_volume,
        width, height, max_disp
    );
    timer.Stop();
    stage_times[1] = timer.ElapsedMs();
    printf("    Time: %.3f ms\n", stage_times[1]);

    // ---- Stage 3: SGM 8-Path Aggregation ----
    printf("  [Stage 3] SGM 8-Path Cost Aggregation...\n");
    timer.Start();

    size_t smem_size = 2 * max_disp * sizeof(uint16_t);
    int total_diags = width + height - 1;

    // Path 1: Left to Right
    sgm_horizontal_kernel<<<height, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, 1);

    // Path 2: Right to Left
    sgm_horizontal_kernel<<<height, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, -1);

    // Path 3: Top to Bottom
    sgm_vertical_kernel<<<width, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, 1);

    // Path 4: Bottom to Top
    sgm_vertical_kernel<<<width, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, -1);

    // Path 5: Top-Left to Bottom-Right
    sgm_diagonal_kernel<<<total_diags, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, 1, 1);

    // Path 6: Bottom-Right to Top-Left
    sgm_diagonal_kernel<<<total_diags, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, -1, -1);

    // Path 7: Bottom-Left to Top-Right
    sgm_diagonal_kernel<<<total_diags, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, 1, -1);

    // Path 8: Top-Right to Bottom-Left
    sgm_diagonal_kernel<<<total_diags, max_disp, smem_size>>>(
        d_cost_volume, d_agg_cost, width, height, max_disp, -1, 1);

    timer.Stop();
    stage_times[2] = timer.ElapsedMs();
    printf("    Time: %.3f ms (8 paths)\n", stage_times[2]);

    // ---- Stage 4: Winner-Takes-All ----
    printf("  [Stage 4] Winner-Takes-All Disparity Selection...\n");
    timer.Start();
    wta_kernel<<<grid2d, block2d>>>(d_agg_cost, d_disparity, width, height, max_disp);
    timer.Stop();
    stage_times[3] = timer.ElapsedMs();
    printf("    Time: %.3f ms\n", stage_times[3]);

    // ---- Stage 5: Median Filter ----
    printf("  [Stage 5] 3x3 Median Filter...\n");
    timer.Start();
    median_filter_kernel<<<grid2d, block2d>>>(d_disparity, d_disparity_filtered, width, height);
    timer.Stop();
    stage_times[4] = timer.ElapsedMs();
    printf("    Time: %.3f ms\n", stage_times[4]);

    // ---- Stage 6: Jet Colormap ----
    printf("  [Stage 6] Jet Colormap Visualization...\n");
    timer.Start();
    jet_colormap_kernel<<<grid2d, block2d>>>(
        d_disparity_filtered, d_color, width, height, max_disp);
    timer.Stop();
    stage_times[5] = timer.ElapsedMs();
    printf("    Time: %.3f ms\n", stage_times[5]);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_disparity, d_disparity_filtered, img_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_color, d_color, color_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_census_left));
    CUDA_CHECK(cudaFree(d_census_right));
    CUDA_CHECK(cudaFree(d_cost_volume));
    CUDA_CHECK(cudaFree(d_agg_cost));
    CUDA_CHECK(cudaFree(d_disparity));
    CUDA_CHECK(cudaFree(d_disparity_filtered));
    CUDA_CHECK(cudaFree(d_color));
}


// ============================================================================
// CPU Pipeline for Benchmarking
// ============================================================================

float run_cpu_pipeline(
    const uint8_t* h_left, const uint8_t* h_right,
    uint8_t* h_disparity_cpu,
    int width, int height, int max_disp
) {
    size_t img_size = (size_t)width * height;

    uint64_t* census_l = (uint64_t*)malloc(img_size * sizeof(uint64_t));
    uint64_t* census_r = (uint64_t*)malloc(img_size * sizeof(uint64_t));
    uint8_t*  cost     = (uint8_t*) malloc(img_size * max_disp);
    uint16_t* agg      = (uint16_t*)malloc(img_size * max_disp * sizeof(uint16_t));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("\n  [CPU] Census Transform...\n");
    cpu_census_transform(h_left,  census_l, width, height);
    cpu_census_transform(h_right, census_r, width, height);

    printf("  [CPU] Cost Computation...\n");
    cpu_compute_cost(census_l, census_r, cost, width, height, max_disp);

    printf("  [CPU] SGM Aggregation (4 paths)...\n");
    cpu_sgm_aggregate(cost, agg, width, height, max_disp);

    printf("  [CPU] Winner-Takes-All...\n");
    cpu_wta(agg, h_disparity_cpu, width, height, max_disp);

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    float elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0f +
                       (t_end.tv_nsec - t_start.tv_nsec) / 1e6f;

    free(census_l);
    free(census_r);
    free(cost);
    free(agg);

    return elapsed_ms;
}


// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s <left.pgm> <right.pgm> [max_disparity]\n", prog);
    printf("  left.pgm       - Left stereo image (PGM P5 format)\n");
    printf("  right.pgm      - Right stereo image (PGM P5 format)\n");
    printf("  max_disparity   - Maximum disparity (default: %d, must be power of 2)\n",
           DEFAULT_MAX_DISP);
}

int main(int argc, char** argv) {
    printf("\n");
    printf("============================================================\n");
    printf("  DepthForge - CUDA-Accelerated Semi-Global Matching\n");
    printf("  Real-Time Stereo Depth Estimation\n");
    printf("============================================================\n");

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* left_file  = argv[1];
    const char* right_file = argv[2];
    int max_disp = (argc >= 4) ? atoi(argv[3]) : DEFAULT_MAX_DISP;

    // Ensure max_disp is power of 2 for reduction kernels
    if (max_disp & (max_disp - 1)) {
        fprintf(stderr, "Error: max_disparity must be a power of 2\n");
        return 1;
    }

    // Print GPU info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\n  GPU: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Global Memory: %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);

    // Load images
    printf("\n  Loading stereo pair...\n");
    int w_left, h_left, w_right, h_right;
    uint8_t* h_left_img  = read_pgm(left_file,  &w_left,  &h_left);
    uint8_t* h_right_img = read_pgm(right_file, &w_right, &h_right);

    if (!h_left_img || !h_right_img) {
        fprintf(stderr, "Error: Failed to load images\n");
        return 1;
    }

    if (w_left != w_right || h_left != h_right) {
        fprintf(stderr, "Error: Image dimensions must match (%dx%d vs %dx%d)\n",
                w_left, h_left, w_right, h_right);
        return 1;
    }

    int width = w_left, height = h_left;
    printf("  Image size: %dx%d\n", width, height);
    printf("  Max disparity: %d\n", max_disp);
    printf("  Census window: %dx%d\n", CENSUS_WIDTH, CENSUS_HEIGHT);
    printf("  SGM penalties: P1=%d, P2=%d\n", SGM_P1, SGM_P2);

    size_t img_size = (size_t)width * height;

    // Allocate output buffers
    uint8_t* h_disparity_gpu = (uint8_t*)malloc(img_size);
    uint8_t* h_disparity_cpu = (uint8_t*)malloc(img_size);
    uint8_t* h_color         = (uint8_t*)malloc(img_size * 3);

    // ---- Run GPU Pipeline ----
    printf("\n=== GPU Pipeline (8 paths) ===\n");
    float gpu_times[6];
    run_gpu_pipeline(h_left_img, h_right_img, h_disparity_gpu, h_color,
                     width, height, max_disp, gpu_times);

    float gpu_total = 0;
    for (int i = 0; i < 6; i++) gpu_total += gpu_times[i];
    float gpu_fps = 1000.0f / gpu_total;

    printf("\n  GPU Total:   %.3f ms (%.1f FPS)\n", gpu_total, gpu_fps);

    // ---- Run CPU Pipeline ----
    printf("\n=== CPU Baseline (4 paths) ===\n");
    float cpu_total = run_cpu_pipeline(h_left_img, h_right_img,
                                       h_disparity_cpu, width, height, max_disp);
    float cpu_fps = 1000.0f / cpu_total;

    printf("\n  CPU Total:   %.3f ms (%.1f FPS)\n", cpu_total, cpu_fps);

    // ---- Speedup Report ----
    printf("\n============================================================\n");
    printf("  BENCHMARK RESULTS\n");
    printf("============================================================\n");
    printf("  Image:         %dx%d (%d disparities)\n", width, height, max_disp);
    printf("  GPU (8-path):  %.3f ms  (%.1f FPS)\n", gpu_total, gpu_fps);
    printf("  CPU (4-path):  %.3f ms  (%.1f FPS)\n", cpu_total, cpu_fps);
    printf("  Speedup:       %.1fx\n", cpu_total / gpu_total);
    printf("  Real-time:     %s (target >= 30 FPS)\n",
           gpu_fps >= 30 ? "YES" : "NO");
    printf("============================================================\n");

    printf("\n  Per-stage GPU timing:\n");
    const char* stage_names[] = {
        "Census Transform", "Cost Computation", "SGM Aggregation (8-path)",
        "Winner-Takes-All", "Median Filter", "Jet Colormap"
    };
    for (int i = 0; i < 6; i++) {
        printf("    %-30s %8.3f ms (%5.1f%%)\n",
               stage_names[i], gpu_times[i], 100.0f * gpu_times[i] / gpu_total);
    }

    // ---- Save Outputs ----
    printf("\n  Saving outputs...\n");
    write_pgm("disparity_gpu.pgm", h_disparity_gpu, width, height);
    write_pgm("disparity_cpu.pgm", h_disparity_cpu, width, height);
    write_ppm("depthmap_color.ppm", h_color, width, height);

    // Also save a normalized version for better visibility
    uint8_t max_d = 0;
    for (size_t i = 0; i < img_size; i++)
        if (h_disparity_gpu[i] > max_d) max_d = h_disparity_gpu[i];

    if (max_d > 0) {
        uint8_t* h_norm = (uint8_t*)malloc(img_size);
        for (size_t i = 0; i < img_size; i++)
            h_norm[i] = (uint8_t)(255.0f * h_disparity_gpu[i] / max_d);
        write_pgm("disparity_normalized.pgm", h_norm, width, height);
        free(h_norm);
    }

    printf("\n  Output files:\n");
    printf("    disparity_gpu.pgm        - GPU disparity map (raw)\n");
    printf("    disparity_cpu.pgm        - CPU disparity map (raw)\n");
    printf("    disparity_normalized.pgm - GPU disparity (contrast-stretched)\n");
    printf("    depthmap_color.ppm       - Jet colormap visualization\n");

    // Cleanup
    free(h_left_img);
    free(h_right_img);
    free(h_disparity_gpu);
    free(h_disparity_cpu);
    free(h_color);

    printf("\n  Done.\n\n");
    return 0;
}
