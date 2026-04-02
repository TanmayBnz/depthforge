# ============================================================================
# DepthForge - Makefile
# ============================================================================
# Build:   make
# Run:     make run  (uses synthetic test data)
# Clean:   make clean
# Profile: make profile
# ============================================================================

NVCC       = nvcc
NVCCFLAGS  = -O3 -std=c++14 --use_fast_math
ARCH       = -arch=sm_75       # Adjust for your GPU (sm_60, sm_70, sm_75, sm_80, sm_86, sm_89)
INCLUDES   =
LIBS       = -lm

# Detect GPU architecture automatically (if nvidia-smi is available)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifneq ($(GPU_ARCH),)
    ARCH = -arch=sm_$(GPU_ARCH)
endif

TARGET     = depthforge
SRC        = src/depthforge.cu

# Debug build
DEBUG_FLAGS = -G -g -lineinfo -DDEBUG

.PHONY: all clean run profile debug test help

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "============================================"
	@echo "  Building DepthForge ($(ARCH))"
	@echo "============================================"
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "  Build successful: ./$(TARGET)"

debug: $(SRC)
	@echo "Building debug version..."
	$(NVCC) $(DEBUG_FLAGS) $(ARCH) $(INCLUDES) -o $(TARGET)_debug $< $(LIBS)

# Generate synthetic test data and run
run: $(TARGET) test_data
	@echo ""
	@echo "============================================"
	@echo "  Running DepthForge"
	@echo "============================================"
	./$(TARGET) data/left.pgm data/right.pgm 128

test_data:
	@echo "Generating synthetic stereo pair..."
	python3 tools/generate_stereo.py --width 640 --height 480 --output data/

# Run with KITTI data (user must download separately)
run_kitti: $(TARGET)
	@echo "Running on KITTI stereo pair..."
	./$(TARGET) data/kitti_left.pgm data/kitti_right.pgm 128

# Profile with nvprof
profile: $(TARGET) test_data
	nvprof --print-gpu-trace ./$(TARGET) data/left.pgm data/right.pgm 128

# Profile with Nsight Compute
ncu: $(TARGET) test_data
	ncu --set full ./$(TARGET) data/left.pgm data/right.pgm 128

clean:
	rm -f $(TARGET) $(TARGET)_debug
	rm -f disparity_gpu.pgm disparity_cpu.pgm disparity_normalized.pgm depthmap_color.ppm

help:
	@echo "DepthForge Build System"
	@echo "========================"
	@echo "  make           - Build the project"
	@echo "  make run       - Build, generate test data, and run"
	@echo "  make debug     - Build with debug symbols"
	@echo "  make profile   - Run with nvprof profiling"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make help      - Show this message"
	@echo ""
	@echo "To change GPU architecture, edit ARCH in Makefile"
	@echo "  e.g., ARCH=-arch=sm_86 for RTX 3080"
