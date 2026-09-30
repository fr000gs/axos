ifndef MAKE_VERSION
$(error This Makefile requires GNU Make)
endif

CACHE_MK := .make-cache.mk

-include $(CACHE_MK)

# nvcc needs old gcc; let the user override,
# otherwise try to find something nvcc is likely to accept.
# Override directly with: make SPIRV_PROFILE=spirv_1_5
$(CACHE_MK):
	@echo "Detecting toolchain..."
	@echo "SLANGC := $$(command -v slangc 2>/dev/null)" > $@
	@echo "NVCC_CCBIN := $$(command -v gcc-14 2>/dev/null || \
	                     command -v gcc-13 2>/dev/null || \
	                     command -v gcc 2>/dev/null)" >> $@
	@echo "SPIRV_PROFILE := spirv_1_3" >> $@
#	@V=$$(vulkaninfo --summary 2>/dev/null | \
#	      grep -m1 -oE 'apiVersion[^0-9]*[0-9]+\.[0-9]+' | \
#	      grep -oE '[0-9]+\.[0-9]+'); \
#	if [ -z "$$V" ]; then \
#		echo "SPIRV_PROFILE := spirv_1_3"; \
#	elif [ "$${V%%.*}" = "1" ] && [ "$${V#*.}" = "3" ]; then \
#		echo "SPIRV_PROFILE := spirv_1_6"; \
#	else \
#		echo "SPIRV_PROFILE := spirv_1_3"; \
#	fi >> $@

#@V=$$(vulkaninfo --summary 2>/dev/null | \
#		grep -m1 -oE 'apiVersion[^0-9]*[0-9]+\.[0-9]+' | \
#		grep -oE '[0-9]+\.[0-9]+'); \
#	MAJOR="$${V%%.*}"; \
#	MINOR="$${V#*.}"; \
#	if [ -z "$$V" ]; then \
#		echo "SPIRV_PROFILE := spirv_1_0"; \
#	elif [ "$$MAJOR" -gt 1 ] || ( [ "$$MAJOR" -eq 1 ] && [ "$$MINOR" -ge 3 ] ); then \
#		echo "SPIRV_PROFILE := spirv_1_6"; \
#	elif [ "$$MAJOR" -eq 1 ] && [ "$$MINOR" -eq 2 ]; then \
#		echo "SPIRV_PROFILE := spirv_1_5"; \
#	elif [ "$$MAJOR" -eq 1 ] && [ "$$MINOR" -eq 1 ]; then \
#		echo "SPIRV_PROFILE := spirv_1_3"; \
#	else \
#		echo "SPIRV_PROFILE := spirv_1_0"; \
#	fi >> $@

configure:
	$(MAKE) $(CACHE_MK)

reconfigure:
	rm -f $(CACHE_MK)
	$(MAKE) configure

# ─── Toolchain ────────────────────────────────────────────────────────────────
CXX  ?= g++
NVCC ?= nvcc

MAKEFLAGS += -j12 --output-sync=target

ifeq ($(SLANGC),)
$(warning slangc not found in PATH. Set SLANGC=/path/to/slangc)
endif
ifeq ($(NVCC_CCBIN),)
$(warning No suitable host compiler found for nvcc. Set NVCC_CCBIN=/path/to/gcc)
endif

# ─── Compiler flags ───────────────────────────────────────────────────────────
CPPFLAGS := -Isrc -I/usr/include/eigen3 -DCUBLAS_WITH -DCUSOLVER_WITH -DCUSPARSE_WITH -DCUDSS_WITH
GCCJIT_CFLAGS := -I/usr/lib/gcc/x86_64-pc-linux-gnu/15/include

CFLAGS := -O3 -march=native
CXXFLAGS := -fopenmp -DPANINI_ENABLE_OMP

# -lvulkan
VULKAN_LDLIBS := $(shell pkg-config --libs vulkan 2>/dev/null || echo -lvulkan)
CUDA_LDLIBS   := $(shell pkg-config --libs cudart 2>/dev/null) -lcublas -lcudnn -lcusolver -lcusparse -lcudss
GCCJIT_LDLIBS := -L/usr/lib/gcc/x86_64-pc-linux-gnu/15 -lgccjit -ldl
GTEST_FLAGS   := $(shell pkg-config --cflags --libs gtest gtest_main 2>/dev/null)

# Linker flags for linking CUDA device code via the host linker.

# ─── NVCC host-compiler flag ──────────────────────────────────────────────────
NVCCFLAGS := $(if $(NVCC_CCBIN),-ccbin=$(NVCC_CCBIN),)

ISA_AVX2 := -mfma -mavx2

# Ensure build/ directories exist before any rule that needs them.
BUILD_DIRS := build build/shaders
$(BUILD_DIRS):
	@mkdir -p $@

# ════════════════════════════════════════════════════════════════════════════════
# System benchmark
# ════════════════════════════════════════════════════════════════════════════════
build/benchmark_sys: benchmarks/benchmark_sys.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

benchmark_sys: build/benchmark_sys
	./build/benchmark_sys

# Autograd benchmark
# ════════════════════════════════════════════════════════════════════════════════
build/benchmark_autograd.o: benchmarks/benchmark_autograd.cpp | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/benchmark_autograd: build/benchmark_autograd.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_autograd: build/benchmark_autograd
	./build/benchmark_autograd

build/benchmark_autograd_cpu: benchmarks/benchmark_autograd.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

benchmark_autograd_cpu: build/benchmark_autograd_cpu
	./build/benchmark_autograd_cpu

# ════════════════════════════════════════════════════════════════════════
# Shaders (Slang -> SPIR-V)
# ════════════════════════════════════════════════════════════════════════
SHADERS_TO_SPECIALIZE := add binary_op scalar_op unary binary_grad scalar_grad unary_grad \
                         sgd_update adam_update adamw_update mse_loss mse_loss_grad \
                         huber_loss_batch huber_loss_batch_grad cross_entropy_loss cross_entropy_grad
TYPES := float half int32 int16 int8

SLANG_SRCS := $(filter-out $(foreach s,$(SHADERS_TO_SPECIALIZE),src/shaders/$(s).slang),$(wildcard src/shaders/*.slang))
SLANG_SPVS := $(patsubst src/shaders/%.slang,build/shaders/%.spv,$(SLANG_SRCS)) \
              $(foreach shader,$(SHADERS_TO_SPECIALIZE),$(foreach type,$(TYPES),build/shaders/$(shader)_$(type).spv))

shaders: $(SLANG_SPVS)

build/shaders/%.spv: src/shaders/%.slang | build/shaders
	$(SLANGC) $< -target spirv -profile $(SPIRV_PROFILE) -entry computeMain -O3 -o $@

entry_float := Float
entry_half  := Half
entry_int32 := Int32
entry_int16 := Int16
entry_int8  := Int8

define SHADER_RULE
build/shaders/$(1)_$(2).spv: src/shaders/$(1).slang | build/shaders
	$$(SLANGC) $$< -target spirv -profile $$(SPIRV_PROFILE) -entry computeMain$(entry_$(2)) -O3 -o $$@
endef

$(foreach shader,$(SHADERS_TO_SPECIALIZE),\
    $(foreach type,$(TYPES),\
        $(eval $(call SHADER_RULE,$(shader),$(type)))\
    )\
)

# Benchmark-specific add shader (lives outside src/shaders/)
build/shaders/add_bench.spv: benchmarks/add_bench.slang | build/shaders
	$(SLANGC) $< -target spirv -profile $(SPIRV_PROFILE) -entry computeMain -O3 -o $@

# ─── Vulkan runtime smoke test ────────────────────────────────────────────────
# vulkanRuntime.cpp is #included directly by test_vk.cpp, so no separate compile step.
build/test_vk: benchmarks/test_vk.cpp build/shaders/add_float.spv \
               src/vulkan_runtime/vulkanRuntime.h | build
	$(CXX) $(CPPFLAGS) -DUSE_VULKAN $(CFLAGS) $(CXXFLAGS) $(VULKAN_LDLIBS) $< -o $@

test_vk: build/test_vk
	./build/test_vk

# ════════════════════════════════════════════════════════════════════════════════
# CUDA addition benchmark (Vulkan + CUDA mixed)
# ════════════════════════════════════════════════════════════════════════════════
build/benchmark_addition.o: benchmarks/benchmark_addition.cu \
                             src/vulkan_runtime/vulkanRuntime.h | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA -DUSE_VULKAN $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" \
	        -x cu -c $< -o $@

build/benchmark_addition: build/benchmark_addition.o build/shaders/add_bench.spv | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(VULKAN_LDLIBS) $(CUDA_LDLIBS) $< -o $@

benchmark_addition: build/benchmark_addition
	./build/benchmark_addition

# ════════════════════════════════════════════════════════════════════════
# CUDA kernels / tests
# ════════════════════════════════════════════════════════════════════════
CUDA_SHADERS := #build/shaders/matmul.o

$(CUDA_SHADERS): build/shaders/%.o: src/shaders/%.cu | build/shaders
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

build/test_cuda.o: tests/test_cuda.cpp tests/test_common.h | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_cuda: $(CUDA_SHADERS) build/test_cuda.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_cuda: build/test_cuda
	./build/test_cuda

# ── test_all (ncurses TUI test runner) ────────────────────────────────────────
build/test_all.o: tests/test_all.cpp tests/test_all.h | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA -DUSE_VULKAN -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_all: $(CUDA_SHADERS) build/test_all.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@ -lncurses -ltinfo

test_all: build/test_all
	./build/test_all --headless

# ── test_cpu (CPU-only, no nvcc) ──────────────────────────────────────────────
build/test_cpu: tests/test_cpu.cpp tests/test_common.h | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) -Itests $< -o $@

test_cpu: build/test_cpu
	./build/test_cpu

# ── test_sparse_cpu (Csr, CPU kernels) ────────────────────────────────────────
build/test_sparse_cpu: tests/test_sparse.cpp tests/test_common.h $(wildcard src/sparse/*.h) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) -Itests $< -o $@

test_sparse_cpu: build/test_sparse_cpu
	./build/test_sparse_cpu

# ── test_sparse (Csr on CPU and CUDA, cuSPARSE/cuDSS) ─────────────────────────
build/test_sparse.o: tests/test_sparse.cpp tests/test_common.h $(wildcard src/sparse/*.h) src/shaders/sparse.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_sparse: build/test_sparse.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_sparse: build/test_sparse
	./build/test_sparse

# ── test_solver (LP/MILP/QP solvers: model, MPS, presolve, PDLP) ─────────────
build/test_solver_cpu: tests/test_solver.cpp tests/test_common.h $(wildcard src/sparse/*.h src/solver/*.h src/solver/*/*.h) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) -Itests $< -o $@

test_solver_cpu: build/test_solver_cpu
	./build/test_solver_cpu

build/test_solver.o: tests/test_solver.cpp tests/test_common.h $(wildcard src/sparse/*.h src/solver/*.h src/solver/*/*.h) src/shaders/sparse.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_solver: build/test_solver.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_solver: build/test_solver
	./build/test_solver

# ── benchmark_lp (Panini PDLP vs HiGHS via SciPy) ────────────────────────────
build/bench_lp.o: benchmarks/solver/bench_lp.cpp $(wildcard src/sparse/*.h src/solver/*.h src/solver/*/*.h) src/shaders/sparse.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/bench_lp: build/bench_lp.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_lp: build/bench_lp
	./build/bench_lp | tee build/bench_lp_panini.txt
	~/python_junk/.venv/bin/python benchmarks/solver/bench_highs.py | tee build/bench_lp_highs.txt

# ── run_mps / benchmark_netlib (real LP instances) ────────────────────────────
build/run_mps.o: benchmarks/solver/run_mps.cpp $(wildcard src/sparse/*.h src/solver/*.h src/solver/*/*.h) src/shaders/sparse.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/run_mps: build/run_mps.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

# ── benchmark_sparse (Panini CPU/CUDA vs Eigen; torch via bench_torch.py) ────
build/bench_sparse.o: benchmarks/sparse/bench_sparse.cpp $(wildcard src/sparse/*.h) src/shaders/sparse.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DPANINI_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/bench_sparse: build/bench_sparse.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_sparse: build/bench_sparse
	./build/bench_sparse | tee build/bench_sparse_panini.txt
	~/python_junk/.venv/bin/python benchmarks/sparse/bench_torch.py | tee build/bench_sparse_torch.txt
	~/python_junk/.venv/bin/python benchmarks/sparse/report.py build/bench_sparse_panini.txt build/bench_sparse_torch.txt

# ── test_gtest (Google Test runner) ──────────────────────────────────────────
# test_gtest.cpp does not exist yet; this target is preserved from the old
# Makefile for when it is added.  Headers live in tests/.
build/test_gtest.o: benchmarks/test_gtest.cpp tests/test_all.h | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_gtest: $(CUDA_SHADERS) build/test_gtest.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@ $(GTEST_FLAGS)

test_gtest: build/test_gtest
	./build/test_gtest

# ════════════════════════════════════════════════════════════════════════════════
# NN benchmark (CPU ×2 + GPU), driven by a Python harness
# Override PYTHON with: make benchmark_nn PYTHON=~/.venv/bin/python3
# ════════════════════════════════════════════════════════════════════════════════
PYTHON ?= python3

build/nn_bench_cpu_no_omp: benchmarks/nn/nn_bench_cpu.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) -ffast-math $< -o $@

build/nn_bench_cpu_omp: benchmarks/nn/nn_bench_cpu.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

build/nn_bench_gpu.o: benchmarks/nn/nn_bench_gpu.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

build/nn_bench_gpu: $(CUDA_SHADERS) build/nn_bench_gpu.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_nn: build/nn_bench_cpu_no_omp build/nn_bench_cpu_omp \
              build/nn_bench_gpu benchmarks/nn/benchmark_nn.py
	$(PYTHON) benchmarks/nn/benchmark_nn.py

# ════════════════════════════════════════════════════════════════════════════════
# Matrix-multiplication benchmark matrix
# ════════════════════════════════════════════════════════════════════════════════
BENCH_NAMES_NATIVE := eigen lazy tiled non_avx gemv_non_avx gemv_eigen \
                      gemv_tiled blocked inverse
BENCH_NAMES_AVX2   := avx2 strassen_avx2 gemv_avx2
BENCH_NAMES_512    := avx512 gemv_avx512

BENCH_EXES_NATIVE_NO_OMP := $(patsubst %,build/bench_%_no_omp,$(BENCH_NAMES_NATIVE))
BENCH_EXES_NATIVE_OMP    := $(patsubst %,build/bench_%_omp,$(BENCH_NAMES_NATIVE))
BENCH_EXES_AVX2_NO_OMP   := $(patsubst %,build/bench_%_no_omp,$(BENCH_NAMES_AVX2))
BENCH_EXES_AVX2_OMP      := $(patsubst %,build/bench_%_omp,$(BENCH_NAMES_AVX2))
BENCH_EXES_512_NO_OMP    := $(patsubst %,build/bench_%_no_omp,$(BENCH_NAMES_512))
BENCH_EXES_512_OMP       := $(patsubst %,build/bench_%_omp,$(BENCH_NAMES_512))

$(BENCH_EXES_NATIVE_NO_OMP): build/bench_%_no_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) -fopenmp-simd $< -o $@

$(BENCH_EXES_NATIVE_OMP): build/bench_%_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

$(BENCH_EXES_AVX2_NO_OMP): build/bench_%_no_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(subst -march=native,$(ISA_AVX2),$(CFLAGS)) -fopenmp-simd $< -o $@

$(BENCH_EXES_AVX2_OMP): build/bench_%_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(subst -march=native,$(ISA_AVX2),$(CFLAGS)) $(CXXFLAGS) $(ISA_AVX2) $< -o $@

$(BENCH_EXES_512_NO_OMP): build/bench_%_no_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) -fopenmp-simd $< -o $@

$(BENCH_EXES_512_OMP): build/bench_%_omp: benchmarks/matops/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

# Single-binary CPU benchmark dispatcher (all implementations compiled in)
build/benchmark_cpu_no_omp: benchmarks/matops/benchmark_cpu.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) -fopenmp-simd $< -o $@

build/benchmark_cpu_omp: benchmarks/matops/benchmark_cpu.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

# GPU matops benchmark
build/benchmark_gpu.o: benchmarks/matops/benchmark_gpu.cu | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

build/benchmark_gpu: $(CUDA_SHADERS) build/benchmark_gpu.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

# Vulkan matops benchmark
build/benchmark_vulkan: benchmarks/matops/benchmark_vulkan.cpp \
                         src/vulkan_runtime/vulkanRuntime.cpp \
                         src/vulkan_runtime/vulkanRuntime.h shaders | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(VULKAN_LDLIBS) \
	       benchmarks/matops/benchmark_vulkan.cpp \
	       src/vulkan_runtime/vulkanRuntime.cpp -o $@

ALL_BENCH_EXES := $(BENCH_EXES_NATIVE_NO_OMP) $(BENCH_EXES_NATIVE_OMP) \
                  $(BENCH_EXES_AVX2_NO_OMP)   $(BENCH_EXES_AVX2_OMP)   \
                  $(BENCH_EXES_512_NO_OMP)     $(BENCH_EXES_512_OMP)    \
                  build/benchmark_cpu_no_omp   build/benchmark_cpu_omp

benchmark_matops: $(ALL_BENCH_EXES) build/benchmark_gpu build/benchmark_vulkan \
                  benchmarks/matops/benchmark_matops.py
	$(PYTHON) benchmarks/matops/benchmark_matops.py

# ════════════════════════════════════════════════════════════════════════════════
# Portable C++ Benchmark Suite (benchmarks/suite/)
# ════════════════════════════════════════════════════════════════════════════════
SUITE_SRC := benchmarks/suite/tp_bench.cpp
SUITE_HDRS := $(wildcard benchmarks/suite/*.h) $(wildcard benchmarks/suite/categories/*.h)

# CPU-only build (no CUDA, no Vulkan)
build/tp_bench: $(SUITE_SRC) $(SUITE_HDRS) | build
	$(CXX) $(CPPFLAGS) -Ibenchmarks/suite $(CFLAGS) $(CXXFLAGS) \
	    -DTP_VERSION=\"$(shell git describe --tags --always 2>/dev/null || echo unknown)\" \
	    $(SUITE_SRC) -o $@

# CUDA-enabled build
build/tp_bench_cuda.o: $(SUITE_SRC) $(SUITE_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Ibenchmarks/suite \
	    $(subst -march=native,,$(CFLAGS)) -Xcompiler "$(CXXFLAGS)" \
	    -DTP_WITH_CUDA \
	    -DTP_VERSION=\"$(shell git describe --tags --always 2>/dev/null || echo unknown)\" \
	    -x cu -c $(SUITE_SRC) -o $@

build/tp_bench_cuda: build/tp_bench_cuda.o | build
	$(CXX) $(CPPFLAGS) -Ibenchmarks/suite $(CFLAGS) $(CXXFLAGS) \
	    -DTP_WITH_CUDA $(CUDA_LDLIBS) $^ -o $@

# Convenience phony: build cpu-only suite and run it
benchmark_suite: build/tp_bench
	./build/tp_bench

# ════════════════════════════════════════════════════════════════════════════════
clean:
	rm -rf build/*

.PHONY: benchmark_lp test_solver test_solver_cpu benchmark_sparse test_sparse test_sparse_cpu configure reconfigure shaders clean \
        benchmark_sys \
        test_vk \
        test_cuda test_cpu test_all test_gtest \
        benchmark_addition benchmark_nn benchmark_matops \
        benchmark_suite

# ── run_milp (branch and bound on MPS files with integer markers; CPU) ────────
build/run_milp: benchmarks/solver/run_milp.cpp $(wildcard src/sparse/*.h src/solver/*.h src/solver/*/*.h) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@
