# tensorPanini

`tensorPanini` is a header-only C++17 tensor library built around **Expression Templates** (ET). CPU code gets lazy, fused evaluation, which avoids temporary arrays and runtime graph traversal. The same `tensorET` type also runs on **CUDA** and **Vulkan** GPUs, and there's a reverse-mode **autograd** engine that works on all three backends.

Current version: **v0.0.4-alpha**. The full API is documented in [ref-manual.txt](ref-manual.txt).

**!NOTE**: Go to stable branch for stable code.

---

## Motivation

Writing numerical code in C or C++ traditionally requires managing raw pointers, writing nested loops for matrix multiplication, and manually tracking memory. This approach is error-prone and provides no compile-time dimensional safety.

**Manual C-style matrix multiplication:**
```c
float* W = (float*)malloc(rows * inner * sizeof(float));
float* X = (float*)malloc(inner * cols * sizeof(float));
float* Y = (float*)malloc(rows * cols * sizeof(float));
// ... populate W, X ...
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        float sum = 0;
        for (size_t k = 0; k < inner; ++k)
            sum += W[i * inner + k] * X[k * cols + j];
        Y[i * cols + j] = sum;
    }
}
// free(W); free(X); free(Y);
```

**The same operation with tensorPanini:**
```cpp
using namespace Panini;
tensorET<2, float> W({rows, inner});
tensorET<2, float> X({inner, cols});
tensorET<2, float> Y = matMul(W, X);   // one line, zero boilerplate
```

`matMul(W, X)` constructs a lazy expression object. No computation occurs until assignment, at which point a single fused pass writes the result directly into `Y`'s backing memory — no intermediate temporaries are allocated. Memory is managed automatically via RAII.

---

## One tensor type, three backends

Since v0.0.4 there is a single tensor class. Where the data lives is chosen by a storage policy:

```cpp
tensorET<2, float>                                   // host memory (default)
tensorET<2, float, Cuda::CudaStorage<float>>         // == tensorETCuda<2, float>
tensorET<2, float, Vulkan::VulkanStorage<float>>     // == tensorETVulkan<2, float>

tensorET<2, float>       h({1024, 1024}, 1.f);
tensorETCuda<2, float>   g = h;   // upload
tensorET<2, float>       back = g; // download
```

Autograd ops dispatch at compile time through `Panini::Cpu::Backend`, `Cuda::Backend` and `Vulkan::Backend` (see `src/backends/`). This means a model written against a generic `TensorType` runs on any of the three backends unchanged. See §4.0 of the ref-manual for how the policies fit together.

---

## Comparison with NumPy and PyTorch

| Operation | NumPy | PyTorch | tensorPanini |
|---|---|---|---|
| Create a matrix | `W = np.zeros((64, 128))` | `W = torch.zeros(64, 128)` | `tensorET<2, float> W({64, 128}, 0.f)` |
| Matrix multiply | `Y = W @ X` | `Y = W @ X` | `Y = matMul(W, X)` |
| Element-wise exp | `np.exp(A)` | `torch.exp(A)` | `exp(A)` |
| Scalar multiply | `A * 2.0` | `A * 2.0` | `A * 2.0` |
| Move to GPU | — | `A.cuda()` | `tensorETCuda<2, float> g = A;` |
| GPU matmul | — | `A.cuda() @ B.cuda()` | `matMulCuda(W, X)` / `matMulVulkan(W, X)` |
| Autograd tensor | — | `torch.tensor(x, requires_grad=True)` | `tensorML<2, float, TensorType>(x, true)` |
| Backward pass | — | `loss.backward()` | `loss.backward()` |
| Zero gradients | — | `optim.zero_grad()` | `optim.zero_grad()` |
| Load an ONNX model | — | — | `Panini::ONNX::load("model.onnx")` |

---

## Features

* **Tensors**: N-dimensional expression-template tensors, slicing/reshape views, blocked tiling, `half`/`_Float16`/`std::complex` element types.
* **CPU**: AVX2/FMA + OpenMP blocked matmul.
* **CUDA**: pooled device allocator, cuBLAS/cuDNN/cuSolver integration.
* **Vulkan**: Slang compute shaders, which also run on Android GPUs.
* **Linear algebra** (CPU): LU (pivoted/tiled), QR (tiled), conjugate gradient, Lanczos, LOBPCG, Padé `expm`. There's also blocked Householder QR on CUDA.
* **Autograd**: conv2d, batch_norm, pooling, linear, activations, MSE/Huber/cross-entropy losses, SGD/Adam/AdamW, a Newton optimizer.
* **Serialization**: raw state dicts, safetensors.
* **Sparse**: CSR matrices with SpMV/SpMM/SpGEMM (cuSPARSE on CUDA) and a sparse LDLᵀ direct solver (cuDSS on CUDA; a multifrontal supernodal code with AMD ordering on the CPU, faster than Eigen's LDLT), the base for the planned LP/MILP/QP solvers (`src/solver/ROADMAP.md`).
* **LP solver**: MPS reader/writer, presolve, and four LP methods: PDLP (reflected Halpern PDHG, CPU and CUDA, with infeasibility detection), a primal-dual interior-point method (cuDSS on the GPU; multifrontal LDL^T or normal equations on the CPU), a CPU dual simplex (hypersparse LU, dual steepest edge, warm starts), and `Auto`, which picks and chains them; see ref-manual §6.9. **MILP**: a branch-and-bound solver on the dual simplex (warm starts, propagation, pseudocosts, diving; no presolve or cuts yet), ref-manual §6.10; `src/solver/ROADMAP.md` has the rest of the plan (QP, MIQP, MINLP).
* **ONNX import**: translates the graph to C++ and JIT-compiles it with libgccjit. This covers a subset of ops (see ref-manual §6.7).
* **MPI** (experimental): collectives, halo exchange, DDP gradient sync, Megatron-style tensor parallelism.

---

## Documentation

* **[ref-manual.txt](ref-manual.txt)**: the complete, current reference (v0.0.4-alpha), with known issues in §0.7.
* `docs/` has topic walkthroughs ([tensors](docs/tensors.md), [CUDA](docs/cuda.md), [Vulkan](docs/vulkan.md), [linear algebra](docs/linear_algebra.md), [autograd](docs/autograd.md), [examples](docs/examples.md)). Parts of the CUDA/Vulkan pages still describe the pre-0.0.4 `tensorETCuda`/`tensorETVulkan` classes and the `.mBuffer` member (now `.storage_.mBuffer`). Where they disagree with the ref-manual, the ref-manual is correct.

---

## Getting Started

### Prerequisites

* **C++17 compiler**: GCC ≥ 9 or Clang ≥ 5 (only GCC 14/15 are regularly tested)
* **OpenMP**: recommended, for parallel CPU matmul
* **CUDA Toolkit** (optional): nvcc, cuBLAS, cuDNN, cuSolver
* **Vulkan SDK + `slangc`** (optional): for the Vulkan backend and shader compilation
* **libgccjit + protobuf** (optional): for ONNX import
* **Raylib** (optional): CartPole visualization
* **Eigen3** (optional): reference results in tests/benchmarks

### Using the library

`tensorPanini` is header-only; point your include path at `src/`:

```cpp
#include "tensorPanini.h"      // CPU tensors + linear algebra
#include "tensorml.h"          // + autograd (tensorML), all backends
#include "tensorCuda.h"        // + tensorETCuda, matMulCuda, ... (compile with nvcc)
#include "tensorVulkan.h"      // + tensorETVulkan, matMulVulkan, ...
#include "tensorml/onnx.h"     // + ONNX import/JIT (link onnx.cpp, jit_compiler.cpp)
```

```bash
# CPU
g++ -std=c++17 -O3 -march=native -fopenmp -Isrc my_program.cpp -o my_program

# CUDA (nvcc needs a host gcc it supports, e.g. gcc-14)
nvcc -std=c++17 -O3 -ccbin=gcc-14 -Isrc -Xcompiler "-O3 -fopenmp" -x cu my_program.cpp \
     -o my_program -lcudart -lcublas -lcudnn

# Vulkan (shaders must be built first: `make shaders`)
g++ -std=c++17 -O3 -march=native -fopenmp -Isrc -DUSE_VULKAN my_program.cpp \
    src/vulkan_runtime/vulkanRuntime.cpp -lvulkan -o my_program
```

---

## Running Tests

```bash
make test_cpu  && ./build/test_cpu     # CPU only
make test_cuda && ./build/test_cuda    # CUDA
make test_vk   && ./build/test_vk      # Vulkan
make test_all  && ./build/test_all     # everything, ncurses TUI
make test_sparse && ./build/test_sparse    # sparse matrices + cuDSS (test_sparse_cpu: CPU only)
make test_solver && ./build/test_solver    # LP solver (test_solver_cpu: CPU only)
```

Add `--headless` (or set `HEADLESS=1`) for plain-text output. The Makefile caches toolchain detection in `.make-cache.mk`. Run `make reconfigure` after installing or changing nvcc/slangc.

---

## Project Structure

```
tensorPanini/
├── ref-manual.txt                  # Full API reference
├── docs/                           # Topic walkthroughs (partly pre-0.0.4)
├── src/
│   ├── tensorPanini.h              # Master include (CPU)
│   ├── tensorBase.h                # CRTP base for expression templates
│   ├── tensorET.h                  # The tensor class (all backends)
│   ├── tensorMath.h                # matmul, blocked/AVX kernels
│   ├── tensorLinearAlgebra.h/.cpp  # Solvers & decompositions
│   ├── tensorCuda.h                # tensorETCuda alias + CUDA helpers
│   ├── tensorVulkan.h              # tensorETVulkan alias + Vulkan ops
│   ├── tensorMPI.h                 # MPI collectives & halo exchange
│   ├── tensorml.h                  # Autograd entry point
│   ├── tensorml/                   # ops, optimizers, safetensors, ONNX/JIT
│   ├── backends/                   # Cpu/Cuda/Vulkan::Backend policies
│   ├── storage/                    # Host/Cuda/Vulkan storage policies
│   ├── tensorcuda/                 # GPU memory pool & library handles
│   ├── vulkan_runtime/             # Vulkan runtime (Apache-2.0)
│   └── shaders/                    # Slang + CUDA kernels
├── examples/
│   ├── cartpole/                   # DQN CartPole (Raylib)
│   ├── mnist/                      # MLP classifier (CPU/CUDA/Vulkan)
│   ├── resnet/                     # ResNet-18 on CIFAR-100 (+ distributed)
│   ├── imagenet/                   # ResNet-50 on ILSVRC
│   ├── model_verifier/             # ONNX JIT vs safetensors check
│   ├── pulse-diffusion/            # 1D/2D diffusion PDE
│   └── tpbinding/                  # Android app, Vulkan inference via JNI
├── benchmarks/
├── tests/
├── LICENSE                         # BSD 3-Clause
└── Makefile
```

---

## License

This project is licensed under the **3-Clause BSD License**; the Vulkan runtime (`src/vulkan_runtime/`) is Apache-2.0. See the [LICENSE](LICENSE) files for details.
