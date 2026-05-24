# AXOS

Sparse linear algebra and optimization solvers in C++17, with CPU (OpenMP) and
CUDA backends.

**Status: this branch does not build.** The dense tensor layer that the sparse
and solver code are written against has been removed. Its interface and
required behaviour, including performance targets, are specified in
[docs/TENSOR_SPEC.md](docs/TENSOR_SPEC.md) and have to be reimplemented first.

## What is here

* **Sparse** (`src/sparse/`): CSR matrices with SpMV/SpMM/SpGEMM/transpose
  (cuSPARSE on CUDA), AMD ordering, and a sparse LDLᵀ direct solver: cuDSS on
  CUDA, and on the CPU a multifrontal supernodal code with packed AVX-512 dense
  kernels.
* **LP solver** (`src/solver/`): MPS reader/writer, presolve with postsolve,
  scaling, and four methods:
  * PDLP: reflected Halpern PDHG, CPU and CUDA, with infeasibility detection.
  * A primal-dual interior-point method: cuDSS on the GPU; multifrontal LDLᵀ or
    normal equations on the CPU; optional crossover.
  * A CPU dual simplex: hypersparse LU, dual steepest edge, bound flipping,
    warm starts.
  * `Auto`, which picks and chains the other three.
* The plan for MILP, QP, MIQP and convex MINLP is in
  [src/solver/ROADMAP.md](src/solver/ROADMAP.md).

## Documentation

* [ref-manual.txt](ref-manual.txt): API of the sparse and LP layers.
* [docs/TENSOR_SPEC.md](docs/TENSOR_SPEC.md): specification of the dense
  tensor layer to be reimplemented.
* [benchmarks/sparse/RESULTS.md](benchmarks/sparse/RESULTS.md),
  [benchmarks/solver/RESULTS.md](benchmarks/solver/RESULTS.md): results
  against Eigen, PyTorch, SciPy and HiGHS (including HiGHS' cuPDLP-C).

## Building (once the tensor layer exists)

Requirements: GNU Make, g++ (C++17, OpenMP), Eigen 3 for the tests. For CUDA:
nvcc with a supported host gcc (`make configure` detects gcc-14/13), cuSPARSE,
cuDSS, cuBLAS and cuSolver.

```sh
make test_sparse_cpu && make test_solver_cpu   # CPU only
make test_sparse && make test_solver           # CPU + CUDA
make benchmark_sparse                          # vs Eigen / torch
make benchmark_lp                              # vs HiGHS (needs scipy/highspy)
make build/run_mps && ./build/run_mps --method auto file.mps
```

## Layout

```
src/sparse/        Csr, CPU/CUDA kernels, AMD, LDLᵀ (multifrontal, cuDSS)
src/shaders/       sparse.cu (CUDA kernels for the sparse layer)
src/solver/        LP model, MPS I/O, presolve, scaling, PDLP, IPM, simplex
tests/             test_sparse.cpp, test_solver.cpp, netlib models in tests/data
benchmarks/        sparse/, solver/ (results + drivers), reference/ (Eigen)
docs/              TENSOR_SPEC.md
```
