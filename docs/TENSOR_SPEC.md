# AXOS dense tensor layer: behavioural specification

Status: the dense tensor layer (`tensorET` and its support headers) has been
removed from this branch. This document specifies what a replacement must do so
that the code that remains (sparse matrices, LP solvers, their tests and
benchmarks) builds and performs as before. It describes interfaces, observable
behaviour and performance targets.

Namespace: everything below lives in `namespace AXOS` (sub-namespaces `AXOS::Cpu`,
`AXOS::Cuda`). Macros use the `AXOS_` prefix.

Contents

1. Scope and priorities
2. Headers and build flags
3. Storage policies (`Cpu::HostStorage`, `Cuda::CudaStorage`)
4. Backend tags and transfer functions
5. The tensor class `tensorET<DIM, T, Storage>`
6. Expressions and element-wise math (Tier 2)
7. Dense linear algebra (Tier 2)
8. Stream output (Tier 2)
9. GPU memory pool and library handles
10. Element types
11. Performance targets
12. Acceptance tests
13. Usage inventory of the retained code
14. Open decisions for the implementers

---

## 1. Scope and priorities

**Tier 1: required to build and run the retained code.** Storage policies,
backend tags and transfer functions, the tensor class for rank 1 and 2 with the
constructors, accessors and conversions listed in §5, and the GPU memory pool
with cuSPARSE and cuDSS handles (§9). Tier 1 is small (see the inventory in §13)
and must be finished first. Until it exists, `make test_sparse` and
`make test_solver` do not compile.

**Tier 2: the general dense library.** Lazy element-wise expressions, math
functions, matrix products, dense factorizations and eigen-solvers, and printing.
The retained code does not use these. They are specified so the library stays a
general tensor library, and they have performance targets against Eigen (§11).

**Out of scope.** The ML stack (autograd, layers, optimizers, ONNX, data loading,
MPI) and the Vulkan backend were removed and are not specified here.

## 2. Headers and build flags

| Header (suggested)          | Provides                                            | Tier |
|-----------------------------|-----------------------------------------------------|------|
| `tensorET.h`                | the tensor class, `Cpu::HostStorage`, `Cpu::Backend`, `backend_of_t` | 1 |
| `tensorCuda.h`              | `Cuda::CudaStorage`, `Cuda::Backend`, `tensorETCuda` alias; includes the pool | 1 |
| `storage/host_storage.h`, `storage/cuda_storage.h` | the storage policies on their own (the sparse code includes these directly) | 1 |
| `tensorcuda/gpu_pool.h`     | `GPUMemoryPool` (§9)                                | 1 |
| `tensorMath.h`, `tensorLinearAlgebra.h`, `tensorIO.h` | Tier 2 functions   | 2 |
| `axos.h`                    | umbrella header: all of the above that apply to the build | 2 |

The retained code includes exactly these names: `tensorET.h`, `tensorCuda.h`,
`storage/host_storage.h`, `storage/cuda_storage.h`, `tensorcuda/gpu_pool.h`.
`tests/test_common.h` includes the umbrella header (it was `tensorPanini.h`; it
must be renamed there).

Build flags (defined by the Makefile, consumed by the headers):

- `AXOS_ENABLE_CUDA`: CUDA storage and backend are available. Without it,
  `tensorCuda.h` need not compile, but `storage/cuda_storage.h` must still
  compile, as a type with the same interface that can never be allocated.
  Host-only builds instantiate templates that mention `Cuda::CudaStorage`
  without using it.
- `CUSPARSE_WITH`, `CUDSS_WITH`, `CUBLAS_WITH`, `CUSOLVER_WITH`: enable the
  corresponding handle getters on the pool (§9).
- `AXOS_ENABLE_OMP`: OpenMP is available (Tier 2 parallel kernels).
- The code is compiled with `-std=c++17` or later, both by g++ and by nvcc
  (host compiler g++-14). Everything in Tier 1 must compile under nvcc.

## 3. Storage policies

A storage policy owns (or borrows) a flat, contiguous array of `n` elements in
one memory space. `Csr<T, Idx, Store>` in `src/sparse/csr.h` uses the policies
directly as `template <typename> class Store` parameters, independently of the
tensor class. So the policies are a public interface in their own right.

### 3.1 Required interface (both policies)

For `template <typename T> struct Policy`:

| Member                             | Behaviour |
|------------------------------------|-----------|
| `using backend_type = <tag>;`      | `Cpu::Backend` for host, `Cuda::Backend` for CUDA (§4). |
| `Policy()`                         | Empty: no memory, `data() == nullptr`, size 0. |
| `explicit Policy(size_t n)`        | Owns `n` elements, **contents unspecified** (not initialized, not touched on the host). `n == 0` gives the empty state. |
| `Policy(size_t n, T x)`            | Owns `n` elements, all equal to `x`. |
| `Policy(T *raw, size_t n, bool own = false)` | Wraps an existing buffer in the policy's memory space. With `own == false` the buffer is never freed by the policy. With `own == true` the policy frees it as if it had allocated it (so it must have come from the same allocator). |
| copy constructor / copy assignment | If the source owns its memory: **deep copy** into a new allocation in the same memory space. If the source is a borrowed wrapper: the copy borrows the same pointer (shallow). Self-assignment is a no-op. |
| move constructor / move assignment | Transfers the pointer and ownership; the source becomes empty and non-owning. `noexcept`. |
| `~Policy()`                        | Frees owned memory; never frees borrowed memory. |
| `void allocate(size_t n)`          | Releases current contents (per ownership), then owns `n` unspecified elements. |
| `T *data()`, `const T *data() const` | Pointer to element 0 in the policy's memory space (a device pointer for CUDA), `nullptr` when empty. |
| public `size_t numElem`, `bool ownmem`, `T *ptr` | Present in the removed version; the retained code does **not** use them, so they are optional. |

Element order is the order of the flat array; there is no padding between
elements. `data()` must be aligned to at least `alignof(T)`. For the CUDA
policy, pooled allocations are 256-byte aligned (§9).

### 3.2 Host policy specifics

- Allocation must not touch memory in `Policy(n)`. Large uninitialized vectors
  are used as scratch in the solvers, and first-touch cost belongs to the caller.
- Allocation failure throws `std::bad_alloc`. The IPM relies on catching it
  for over-large factorizations.

### 3.3 CUDA policy specifics

- Memory comes from `GPUMemoryPool::get()` (§9) and is returned to it on
  destruction. There is no `cudaMalloc`/`cudaFree` per object in steady state.
- `Policy(n, x)` fills on the device. Any method is acceptable, but for `x`
  equal to zero it must cost no more than one `cudaMemset`-equivalent pass.
- Deep copies are device-to-device.
- Constructors and assignments are synchronous with respect to the default stream:
  when they return, the data is in place.
- Must compile in translation units that are not compiled by nvcc (plain g++
  with `AXOS_ENABLE_CUDA`), because host code declares CUDA vectors.

## 4. Backend tags and transfer functions

`Cpu::Backend` and `Cuda::Backend` are structs used as tags (template
arguments of `Parallel<Backend>` and `Sparse::Kernels<Backend>` in the retained
code) and as holders of static transfer functions:

```
template <class Tensor>          static T    read_element(const Tensor &t, size_t i);
template <class Tensor>          static void copy_data(Tensor &dst, const Tensor &src);   // same size, same space
template <class Tensor, class U> static void copy_to_host(const Tensor &t, U *host_dst); // t.size() elements
template <class Tensor, class U> static void copy_from_host(Tensor &t, const U *host_src);
```

For the host tag these are plain copies. For CUDA they are blocking copies of
exactly `t.size()` elements. `read_element` returns one element by value, so on
CUDA it is a one-element device-to-host transfer.

`backend_of_t<X>` is `typename X::backend_type` when `X` has one, otherwise
`Cpu::Backend`.

`is_host_storage<S>` is a trait that is true exactly when
`S::backend_type` is `Cpu::Backend`.

## 5. The tensor class

```
template <int DIM, typename T, typename Storage = Cpu::HostStorage<T>>
class tensorET;

template <int DIM, typename T>
using tensorETCuda = tensorET<DIM, T, Cuda::CudaStorage<T>>;   // tensorCuda.h
```

A `tensorET` is a dense, **row-major**, rank-`DIM` array whose elements live in
the memory of `Storage`. `DIM >= 1`. Ranks 1 and 2 are Tier 1; higher ranks
are Tier 2.

### 5.1 Member types and constants

`value_type` and `data_type` (= `T`), `storage_type` (= `Storage`),
`backend_type` (= `Storage::backend_type`), `static constexpr int rank = DIM`.

### 5.2 Required public data

- `T *data`: a **public data member** (not a function) pointing at element 0,
  in the storage's memory space. The retained code reads and writes through
  `x.data` in host loops, passes it to CUDA kernels, and passes it to cuSPARSE and
  cuDSS. It must equal `storage_.data()` for an owning tensor, and it may be
  offset into another tensor's buffer for views.
- `Storage storage_`: public, the owning/borrowing policy object.

Other bookkeeping (sizes, strides, ownership flags) may be public or private at
the implementers' choice. The retained code accesses only `data` and
`storage_`, and the latter only in documentation.

### 5.3 Constructors

| Signature | Behaviour | Tier |
|-----------|-----------|------|
| `tensorET()` | Empty: `size() == 0`, `data == nullptr`, every `size(d) == 0`. | 1 |
| `tensorET(std::initializer_list<size_t> dims)` | Shape `dims`, contents unspecified. Throws `std::runtime_error` if `dims.size() != DIM`. | 1 |
| `tensorET(std::initializer_list<size_t> dims, T x)` | Same, every element equal to `x`. | 1 |
| `tensorET(size_t n)` | `n` elements, contents unspecified. For `DIM == 1` shape `{n}`; for `DIM > 1` shape `{1, ..., 1, n}` (the size goes to the **last** dimension). | 1 |
| `tensorET(size_t n, T x)` | Same, filled with `x`. | 1 |
| `tensorET(size_t dims[DIM])`, `tensorET(size_t dims[DIM], T x)` | Array forms of the initializer-list constructors. | 1 |
| `tensorET(T *raw, size_t dims[DIM])` | Non-owning view of an existing buffer in `Storage`'s memory space; never frees it. | 1 |
| `tensorET(T *raw, size_t dims[DIM], bool own)` | As above; with `own == true` the tensor frees the buffer through `Storage`. | 2 |
| `tensorET(Storage &&s, size_t dims[DIM])` | Adopts an existing storage object; `s.size` must be the product of `dims` (undefined otherwise). | 2 |
| `tensorET(size_t dims[DIM], size_t start[DIM], size_t end[DIM])` | Shape `dims`, and additionally records a sub-domain `[start, end]` (inclusive) per dimension that can be read back (§5.7). Contents unspecified. | 2 |
| copy constructor / assignment | Same semantics as the storage policy: owning tensors deep-copy; **views stay views** (the copy aliases the same memory). | 1 |
| move constructor / assignment | Steals the buffer; the source becomes empty (`size() == 0`, `data == nullptr`). `noexcept`. | 1 |
| `template <class S2> tensorET(const tensorET<DIM, T, S2> &o)` and `operator=(const tensorET<DIM, T, S2> &)` | **Cross-storage conversion.** Allocates a new owning tensor of `o`'s shape in *this* storage and copies all elements across memory spaces with the backend transfer functions (§4). Host↔host is a straight copy; host↔device is one transfer; device↔device through host staging is acceptable. After assignment the target's previous contents are released. | 1 |
| `template <class E> tensorET(const Expr<T, E> &e)` | Host only: allocates the expression's shape and evaluates it element by element (§6). | 2 |

Host-to-device and device-to-host conversions are how the solvers upload and download
vectors (`Vec d(h)` / `tensorET<1,double> h(d)`). They must be one blocking copy
and nothing more (see the §11 targets).

### 5.4 Size queries

- `size_t size() const`: total element count (product of the shape).
- `size_t size(int d) const`: extent of dimension `d`, for `0 <= d < DIM`.
  Throws `std::runtime_error` otherwise.
- `size_t stride(int d) const`: element stride of dimension `d` (row-major:
  the last dimension has stride 1). Throws `std::runtime_error` for out-of-range `d`.
  For views created by `slice`, the strides are the **parent's** strides.

### 5.5 Element access

Host storage:

- `T &operator[](size_t i)` and `const T` overload: flat element `i` of the
  underlying buffer (no bounds check). For contiguous tensors this is the
  row-major linear index.
- `T &value(size_t i)` and `T value(size_t i) const`: same as `operator[]`.
- `T &operator()(i0, ..., i_{DIM-1})` and const overload: element at the
  coordinates, computed from the strides. A call with the wrong number of indices
  must fail to compile.

Non-host storage (CUDA):

- Only the **const, by-value** forms exist: `operator[](i)`, `value(i)` and
  `operator()(...)` return a copy of the element via `read_element`. Writable
  element access must fail to compile. This is intentional: silent per-element
  device round trips in loops are a performance bug.

### 5.6 Views

Views never own memory and are valid only while the tensor they came from
lives. They keep the source's `Storage` type.

- `slice(std::array<size_t,DIM> start, std::array<size_t,DIM> length)`: view
  of the box `[start, start+length)`. Shape is `length`, strides are the parent's,
  and `data` points at the first element of the box. No bounds checking.
- `template <int N> reshape(std::array<size_t,N> dims)`: rank-`N` view of the
  same contiguous buffer with fresh row-major strides. Throws
  `std::runtime_error` if the product of `dims` differs from `size()`. It is
  only meaningful on contiguous tensors (not on slices).
- `flatten()`: `reshape<1>({size()})`.

(Tier 2. Slice and reshape work on CUDA tensors as well, since they only move the
pointer.)

### 5.7 Domain

`getIndex()` returns an object from which, for each dimension, the extent, a
start and an end index can be read. By default start is 0 and end is extent-1.
With the domain constructor they are the given values. Tier 2; the retained code
does not use it.

### 5.8 2-D tiling helpers (Tier 2, host only)

For `DIM == 2`, with square tiles of size `m`, where tile `(i, j)` covers rows
`[i*m, i*m+m)` and columns `[j*m, j*m+m)` clipped to the matrix:

- `blockView(i, j, m)`: `slice({i*m, j*m}, {m, m})` (unclipped; caller
  ensures the tile is interior).
- `blockCopy(i, j, m, T *buf, bool transpose = false)`: writes the tile into the
  caller's `m*m` row-major buffer. Parts outside the matrix are **zero**. With
  `transpose`, it writes the transpose of tile `(j, i)`. Returns a non-owning
  `m x m` view of `buf`.
- `blockCopy(i, j, m)`: returns a new owning `m x m` tile, zero-padded.
- `blockWriteBack(i, j, m, tile)`: copies the in-range part of `tile` into the
  matrix (the padding is ignored).
- `blockAddWriteBack(i, j, m, tile)`: same with `+=`.

CUDA equivalents (free functions in `tensorCuda.h`, all on a given
`cudaStream_t`, asynchronous with respect to the host) are
`blockCopyCuda(t, i, j, m, buf, stream)` (non-owning view of `buf`, zero-padded),
`blockCopyCuda(t, i, j, m, stream)` (owning tile), and
`blockWriteBackCuda(t, i, j, m, tile, stream)`.

### 5.9 Random fill (Tier 2, host)

`static tensorET uniform(dims, lo = 0, hi = 1)` and
`static tensorET gaussian(dims, mean = 0, stddev = 1)` return owning host tensors
of i.i.d. samples. Seeding is non-deterministic. Adding an optional seed
parameter is recommended; tests must not depend on the values.

### 5.10 Assignment from a scalar sequence (Tier 2)

The removed version allowed assigning a "place holder" object that fills the
first `N` elements with a constant. It is rarely used. Treat it as optional.

## 6. Expressions and element-wise math (Tier 2)

All element-wise operations are **lazy**. An expression over tensors of equal
shape is an object that can report `size(d)` and compute `value(i)` for any flat
index. It is only evaluated when assigned to a tensor. A chain such as
`C = exp(sin(A) + cos(A)) * 2.0 + B` must run as **one pass** over the data
with **no temporary tensors**. The fusion is required for the targets in §11.

Operations:

- binary, tensor/expression ⊕ tensor/expression, element-wise: `+ - * /`
  (`*` is the element-wise product, not a matrix product);
- scalar, both sides: `x * s`, `s * x`, `x + s`, `s + x`, `x - s`, `s - x`,
  `x / s`, `s / x`;
- unary functions: `sqrt, exp, Log (natural log), sin, cos, tan, sinh, cosh,
  tanh, fabs`;
- `tensorET = expression` (host) evaluates into the **existing** buffer. Sizes
  must match (undefined otherwise; a debug-mode check is recommended). A
  tensor constructed from an expression allocates first.

Users can write their own expression types. An expression type must provide
`value(size_t i)` and `size(int d)` and derive from the library's expression
base so that the operators above accept it. Document the base class's name and
requirements in the new headers.

Matrix products:

- `matMul(A, B)`: lazy product `A (m x k) * B (k x n)`; `value(i)` computes
  entry `(i / n, i % n)`.
- `matMulTile(A, B)`: eager, cache-tiled product returning an `m x n` tensor.
- `matMulBlocked(A, B, C, m, alpha = 1, const tensorET *bias = nullptr, beta = 1,
  bool transA = false, bool transB = false, bool accumulate = false)`, which computes
  `C = alpha * op(A) op(B) + beta * bias` with `op` an optional transpose. With
  `accumulate`, the product is added to `C` instead of overwriting it. `bias`
  (if given) has `C`'s shape. `m` is a tile-size hint. OpenMP-parallel.
- `matMulND(A, B)`: batched product over leading dimensions (rank ≥ 3): the
  last two dimensions multiply, the leading ones must match.
- `extractSlice(t, b)` and `insertSlice(t, b, M)`: get and put the `b`-th 2-D
  matrix of a rank ≥ 3 tensor, with the leading dimensions flattened in row-major
  order.
- `getValue(t, idx*)` and `setValue(t, idx*, v)`: element access by an index array.

## 7. Dense linear algebra (Tier 2, host, `tensorLinearAlgebra.h`)

Matrices are `tensorET<2,T>` and vectors are `tensorET<1,T>`. Errors
(singular pivot, shape mismatch) throw `std::runtime_error`.

| Function | Contract |
|----------|----------|
| `norm(A, B, int p)` | p-norm of `A - B` over all elements (`p >= 1`). |
| `norm(A, B, "fro"/"inf"/...)` | named norms of `A - B`. |
| `conjugateGradient(A, b, x0, maxIter = -1, tol = 1e-6)` | CG for SPD `A`. `maxIter < 0` means `n`. Stops when the residual norm is at most `tol` (absolute; document if relative). Returns `x`. |
| `gaussJordanElimination(A)` | Reduced row-echelon form of `A` (partial pivoting). |
| `augmentMatrix(A, b)` | `[A | b]`. |
| `gramSchmidtOrthogonalization(V)` | Matrix whose columns are an orthonormal basis of `V`'s columns, in order. |
| `luDcmp(A, tol = 1e-12)` | `{L, U}` without pivoting, `A = L U`, unit-diagonal `L`. Throws if a pivot has absolute value at most `tol`. |
| `luDcmpPivoted(A, tol)` | `{{L, U}, P}` with `P A = L U`, `P` a permutation matrix. |
| `luDcmpPivotedTile(A)` | Same result, tiled/parallel algorithm for large `n`. |
| `qrDecompositionTile(A)` | QR (Householder, blocked). Document the return convention (the removed version returned the factored matrix in place). |
| `luSolve(L, U, P, B)` | Solves `A X = B` for a matrix or vector `B`, given the factors of `luDcmpPivoted`. |
| `lanczos(A, m, q0)` | Returns `{alpha (m), beta (m-1), Q (m x n)}`: tridiagonal coefficients and Krylov basis (with reorthogonalization) for symmetric `A`, starting at `q0`. |
| `expm(A)` | Matrix exponential (scaling and squaring with a Padé approximant, or equivalent accuracy: relative error at most 1e-12 on well-conditioned inputs). |
| `lobpcg(A, nev, X0, eigvals, X, maxIter = 100, tol = 1e-8)` | The `nev` smallest eigenpairs of symmetric `A` from the initial block `X0`. Writes `eigvals` (ascending) and `X` (columns = eigenvectors). |
| `elimStep`, `revEl`, `matMul(history, A)`, `matMul(A, history)`, `inverse_backs(A, m = 4)` | A recorded-elimination matrix inverse. The elimination runs once as a sequence of row operations and permutations (`Op` = tagged union of `ElimOp{target_row, source_row, alpha}` and `PermOp{target_row, source_row}`), and the recorded sequence can be replayed on another matrix from the left or the right. `inverse_backs` returns `A^{-1}` by replaying the history on the identity. The pairwise ("binary") elimination order is what makes it parallel. `m` is a block-size hint. |

Accuracy requirement for the factorizations and solves: on random
well-conditioned matrices (condition number up to 1e3, `n` up to 2000), relative
residuals below `1e-12 * n` in double precision.

## 8. Stream output (Tier 2)

`std::ostream &operator<<(std::ostream&, const tensorET<DIM,T>&)` prints nested
brackets, one level per dimension, with the innermost elements separated by two
spaces and each sub-array on its own line, indented by two spaces per level. It
is for debugging only, not a serialization format.

## 9. GPU memory pool and library handles (Tier 1 with CUDA)

`GPUMemoryPool` (in `tensorcuda/gpu_pool.h`) is a per-thread singleton
obtained with `GPUMemoryPool::get()`. The retained code calls:

- `void *allocate(size_t bytes)`: returns device memory of at least `bytes`,
  aligned to 256 bytes; `nullptr` for `bytes == 0`. Freed blocks are reused for
  later requests that fit, preferring the smallest block that fits, so a loop
  that repeatedly allocates the same sizes performs no `cudaMalloc` after the
  first iteration. If the device is out of memory, the pool first returns all
  cached blocks to the driver and retries once. If that fails, it throws
  `std::runtime_error` with a message that includes the MB currently held and the
  requested size.
- `void deallocate(void *p, size_t bytes)`: returns a block to the pool
  (`bytes` is the size originally requested); no-op for `nullptr`.
- `void clear()`: returns all cached blocks to the driver.
- `cusparseHandle_t get_cusparse()` (with `CUSPARSE_WITH`) and
  `cudssHandle_t get_cudss()` (with `CUDSS_WITH`): lazily created, one per
  thread, reused for the thread's lifetime. They throw `std::runtime_error` if
  creation fails. `get_cublas()` and `get_cusolver()` follow the same pattern
  under their flags.
- Allocation mode is a compile-time choice: `Pool` (default, described above),
  `Raw` (direct `cudaMalloc`/`cudaFree` per call), `Async`
  (`cudaMallocAsync`/`cudaFreeAsync` on the default stream). Only `Pool` is
  required.

Handles and memory may be leaked at process exit (destruction order with the
CUDA runtime is not guaranteed), but not during normal operation.

## 10. Element types

Tier 1 needs `double`, `float`, `int32_t` and `int64_t` in both storage
policies (the sparse code stores indices in them). Tier 2 must also accept
`std::complex<float|double>` and a half-precision type: the compiler's
`_Float16` where available, with a 16-bit fallback struct convertible to and
from `float` otherwise, and CUDA's `__half` under nvcc. Other element types
should be rejected at compile time.

## 11. Performance targets

Machine for all targets: the development laptop (AMD Ryzen 7 7840HS, 8 cores /
16 threads, AVX-512; RTX 4060 Laptop GPU, CUDA 12.9), **on AC power**, g++
`-O3 -march=native -fopenmp`, double precision unless noted. Report the median of at
least 5 runs after a warm-up.

### 11.1 Tier 1: no regression of the retained code

The tensor layer sits under every solver vector operation, so the end-to-end
benchmarks are the real test. With the new implementation:

- `make benchmark_sparse`: every row of the table in
  `benchmarks/sparse/RESULTS.md` within **±5 %** of the recorded value (for example
  `spmv rand1m` CPU 3.66 ms, Eigen 4.71 ms; `ldl_factor lap200` CPU 12.5 ms;
  cuDSS 3.6 ms).
- `build/run_mps` on the netlib models and `build/bench_lp`: times within
  **±5 %** of `benchmarks/solver/RESULTS.md` (for example IPM 25fv47 about 55 ms
  CPU; PDLP CUDA transport700 at 1e-6 about 0.96 s, mcf50k about 0.93 s,
  packing200k about 5.1 s; HiGHS cuPDLP-C 1.8 / 1.9 / 19.3 s).
- Iteration counts **identical** to the recorded ones (the container must not
  change any arithmetic).

Micro-targets (these explain regressions when the above misses):

| Operation | Target |
|-----------|--------|
| `.data` access, `size()` | compile to a load; no call, no branch |
| `tensorET<1,double> v({n})` (uninitialized, host) | O(1) apart from the allocator; no page touched |
| `tensorET<1,double> v({n}, 0.0)`, n = 1e7 | ≤ 1.05 × `std::vector<double>(n, 0.0)` |
| host deep copy, n = 1e7 | ≤ 1.05 × `std::memcpy` |
| host→device / device→host conversion, n = 1e7 doubles | ≥ 90 % of a bare `cudaMemcpy` of the same pageable buffer |
| `Cuda` `({n}, 0.0)`, n = 1e7 | ≤ 1.1 × `cudaMemset` of the same size + launch |
| pool: `allocate`+`deallocate` of a previously seen size | ≤ 2 µs host time, zero `cudaMalloc` calls (check with `nsys`) |
| `read_element` on CUDA | one 8-byte transfer (≈10 µs); documented as slow |

### 11.2 Tier 2: against Eigen 3.4

Eigen with `-O3 -march=native`; with OpenMP where Eigen parallelizes (GEMM) and
also without it. Reference harnesses that do not use the removed code are kept in
`benchmarks/reference/` (`bench_eigen.cpp` for GEMM, `bench_gemv_eigen.cpp` for
GEMV). Percentages are throughput relative to Eigen: 100 % means equal time, and
higher is better.

| Operation (sizes) | Target | Minimum acceptable |
|-------------------|--------|--------------------|
| fused element-wise `C = A + B * 2.0`, n = 1e6 … 1e8, 1 thread | 100 % | 90 % |
| fused unary chain `C = exp(sin(A) + cos(A))`, n = 1e6, 1 thread | 100 % (same libm) | 85 % |
| same, OpenMP (Eigen single-threaded) | ≥ 400 % | 250 % |
| `matMulBlocked` GEMM, n = 1024 … 4096, OpenMP both | 90 % | 70 % |
| `matMulBlocked` GEMM, n = 256, OpenMP both | 70 % | 50 % |
| GEMM, 1 thread, n = 1024 | 90 % | 70 % |
| GEMV (`matMulBlocked` with a 1-column B, or a dedicated routine), n = 4096 | 90 % | 75 % |
| `luDcmpPivoted`/`luDcmpPivotedTile` vs `PartialPivLU`, n = 1000 … 4000 | 70 % | 40 % |
| `luSolve`, one RHS, n = 2000 | 90 % | 60 % |
| `inverse_backs` vs `Eigen::inverse()` (OpenMP both), n = 1024 … 4096 | 100 % | 60 % |
| `qrDecompositionTile` vs `HouseholderQR`, n = 1000 … 2000 | 70 % | 40 % |
| `conjugateGradient` per iteration, n = 2000 dense SPD | 90 % of Eigen `ConjugateGradient` on the same dense matrix | 70 % |
| `expm` vs `Eigen::MatrixExponential` (unsupported module), n = 200 | 80 % | 50 % |

Single-thread dense kernels should reach at least 60 % of the core's FP64
peak for GEMM (AVX-512 FMA; the CPU LDLᵀ kernel in `src/sparse/dense_ldl.h` shows about
49 GFLOPS per core is achievable on this machine).

Baseline for the removed implementation (to be filled in on AC power by
building the benchmark harnesses at commit `03a4c03`, branch `feature/csr`,
where the old code still exists; this records where the old code stood and does
not raise the targets above):

| Operation | old / Eigen | measured on |
|-----------|-------------|-------------|
| element-wise fused (n = 1e7) | _pending_ | |
| GEMM n = 2048 OpenMP | _pending_ | |
| GEMV n = 4096 | _pending_ | |
| pivoted LU n = 2000 | _pending_ | |
| inverse_backs n = 2048 | _pending_ | |

## 12. Acceptance tests

1. `make test_sparse_cpu` and `make test_solver_cpu` compile and pass (last
   recorded: 630 and 268 checks). With CUDA, `make test_sparse` and
   `make test_solver` pass (1176 and 330 checks).
2. The §11.1 benchmarks are within tolerance and the iteration counts are identical.
3. A new tensor test suite (to be written with the implementation) covering
   every row of §5, including:
   - shape errors throw (`{..}` with the wrong count, `size(DIM)`, `stride(-1)`, bad `reshape`);
   - copying a view aliases, and copying an owning tensor does not (mutate one, check the other);
   - after a move, the source is empty and the destination holds the data;
   - host→device→host round trip is bit-identical for all Tier 1 element types, including n = 0 and n = 1;
   - `slice` strides equal the parent's; `reshape` of a slice is not required to work;
   - `blockCopy` zero padding at the right and bottom edges; `transpose=true`;
     write-back and add-write-back touch only in-range elements;
   - CUDA writable element access does not compile (a negative compile test, or a
     `static_assert` check via detection idiom);
   - the pool reuses blocks (allocate/free/allocate same size returns a block
     without a new `cudaMalloc`) and recovers from a simulated OOM.
4. Tier 2: each function in §6–§7 against an Eigen reference: element-wise results
   bit-identical to the same scalar expression; factorizations to the §7 accuracy;
   expression chains produce no heap allocation (check with a counting allocator).

## 13. Usage inventory of the retained code

What the kept files actually use, so Tier 1 can be sized precisely:

- Types: `tensorET<1, T, S>` and `tensorET<2, T, S>` for `T` in {double, float,
  int32_t, int64_t}, `S` in {`Cpu::HostStorage<T>`, `Cuda::CudaStorage<T>`};
  `Cpu::Backend`, `Cuda::Backend` as tags; `value_type`, `backend_type`.
- Constructors: `({n}, x)`, `({m, k}, x)`, cross-storage `Vec d(host)` /
  `tensorET<1,T> h(device)`; copy and move of vectors (members of solver objects).
- Members: `.data` (field), `.size()`, `.size(0)`, `.size(1)`.
- Storage policies used directly by `Csr`: `Store<T>(n)`, `.data()`, copy,
  move, default construction.
- `GPUMemoryPool::get().allocate/deallocate/get_cusparse/get_cudss`.
- Macros: `AXOS_ENABLE_CUDA`, `CUSPARSE_WITH`, `CUDSS_WITH` (Makefile);
  `AXOS_HD` is defined by the solver itself in `pdlp_kernels.h`.

Nothing else in §5–§10 is used by the retained code.

## 14. Open decisions for the implementers

- Names: `tensorET` and the header names are kept so that the retained code
  compiles unchanged. Renaming them (like the namespace was) is a mechanical
  follow-up.
- Whether bookkeeping fields stay public (§5.2).
- Seeds for `uniform`/`gaussian` (§5.9).
- Whether the placeholder assignment (§5.10) and the domain constructor
  (§5.3, §5.7) are worth keeping.
- `qrDecompositionTile`'s return convention (§7).
