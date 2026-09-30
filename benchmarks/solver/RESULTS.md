# LP solver benchmark results

Panini's LP solvers (`solve_lp`: presolve + scaling on) against HiGHS on
(1) generated structured LPs and (2) real netlib models. Machine: 16
hardware threads, RTX 4060 Laptop GPU, on AC power, double precision. HiGHS
is the version bundled with SciPy 1.18.1 (generated instances, run through
`scipy.optimize.linprog`) and highspy 1.15.1 (netlib references), at its
default tolerances (about 1e-7). Times are wall-clock ms including presolve
and scaling. `(limit)` is a time/iteration limit, `(killed)` a HiGHS IPM
solve that ignored its time limit and was killed after 120 s.

Methods: `pdlp_*_1e-4` / `1e-6` are PDLP at those relative tolerances;
`ipm_*_1e-8` is the interior-point method at 1e-8; `auto_cpu` is
interior point first with PDLP as fallback (tolerance 1e-6).

Regenerate: `make benchmark_lp` (generated instances, exported to
`build/lp_bench` so both sides solve identical problems; `report.py`
prints the table) and, for netlib, `benchmarks/solver/fetch_netlib.sh`,
`build/run_mps`, `netlib_reference.py`, `netlib_report.py`.

## 1. Generated instances

`transport100/300/700` (dense transportation problems), `mcf10k/50k`
(min-cost flow on a random graph), `packing20k/200k` (sparse packing
LPs; 200k has 2M nonzeros). PDLP and HiGHS were measured before the
kernel fusion was re-timed; the interior-point rows were only run where it
finishes in reasonable time.

```
instance           pdlp_cpu_1e-4    pdlp_cuda_1e-4     pdlp_cpu_1e-6    pdlp_cuda_1e-6      ipm_cpu_1e-8     ipm_cuda_1e-8        highs_auto          highs_ds         highs_ipm
mcf10k                       112                60               264               178    121575 (limit)              8855               444               437               805
mcf50k                      1028               226              5301               842                 -                 -              4976              4654              6008
packing20k                    68                72               996               565    343346 (limit)    126350 (limit)     60106 (limit)     60046 (limit)     61287 (limit)
packing200k                 5718              1097             33101              4825                 -                 -     60381 (limit)     60368 (limit)   120000 (killed)
transport100                  20               246                24                24                24               308                37                37                62
transport300                 159               150               361               169               489               419               262               266               519
transport700                8054               925              6669               934              5627              3386              1874              1862              3369

relative objective difference vs HiGHS (dual simplex / IPM optimum):
instance         pdlp_cpu_1e-4  pdlp_cuda_1e-4   pdlp_cpu_1e-6  pdlp_cuda_1e-6    ipm_cpu_1e-8   ipm_cuda_1e-8
mcf10k                 1.1e-06         1.3e-06         5.8e-08         5.1e-08               -         0.0e+00
mcf50k                 1.5e-06         1.5e-06         1.4e-08         1.4e-08               -               -
packing20k                   -               -               -               -               -               -
packing200k                  -               -               -               -               -               -
transport100           1.4e-06         2.9e-06         4.7e-08         1.5e-07         5.3e-10         5.3e-10
transport300           2.1e-06         6.6e-06         1.2e-08         9.1e-08         0.0e+00         0.0e+00
transport700           3.6e-08         1.8e-07         2.0e-09         3.5e-08         4.0e-10         4.0e-10
```

- Accuracy: PDLP agrees with HiGHS to about 1e-6 relative at 1e-4 and
  1e-8..1e-7 at 1e-6; the interior-point method to 1e-9.
- Packing LPs: neither HiGHS (simplex or IPM) nor a direct-factorization
  interior-point method finishes packing20k, while PDLP solves it in
  0.07-1 s and the 2M-nonzero one in 1-5 s on the GPU. This is the regime
  first-order methods are for.
- Network / transportation LPs: on the GPU PDLP is 2-6x faster than HiGHS
  from about 500k nonzeros (mcf50k: 0.84 s vs 4.7 s; transport700: 0.93 s
  vs 1.9 s); on the CPU HiGHS' simplex wins on transport700 (1.9 s vs 6.7 s).
- Interior point: fine on structured problems (transport700: 5.6 s CPU,
  3.4 s GPU, about HiGHS' own IPM at 3.4 s), but on random sparse graphs
  the KKT factors fill in almost completely: mcf10k takes 7 iterations in
  121 s on the CPU and 9 s on the GPU (HiGHS IPM: 0.8 s with a different,
  iterative method), packing20k does not finish. Sparse factorization
  speed is the limiting factor (see the ROADMAP).
- Small problems (< 100k nonzeros): CPU and GPU take about the same time;
  the GPU pays a one-time context startup (the 246 ms in the first row).

## 2. Netlib (25 models from the HiGHS repository)

19 are feasible (`gt2` is an integer model; its LP relaxation is used) and
6 are netlib's deliberately infeasible ones (bgetam, forest6, galenet,
klein1, refinery, vol1). Limits: 30 s (PDLP), 60 s (IPM), 20 s (Auto).

```
model         HiGHS    verdict |         auto_cpu          ipm_cpu         ipm_cuda         pdlp_cpu        pdlp_cuda
25fv47        104.1    optimal |             51.8             55.3            377.5           1039.8           2979.0
80bau3b        79.3    optimal |            139.8            111.8            284.2           9903.9           6610.9
adlittle        0.9    optimal |              0.6              0.6             15.4             17.0            406.9
afiro           0.3    optimal |              0.2              0.2             12.0              0.7             14.6
avgas           0.1    optimal |              0.2              0.2             11.7              0.2              5.0
bgetam          1.7 infeasible |    20084.2 limit       79.0 limit     326.8 numerr    22551.0 limit    30002.4 limit
blending        0.1    optimal |              0.1              0.1              8.6              0.3              5.1
e226            4.5    optimal |              4.4              4.5             37.1            155.7           1267.2
etamacro        5.8    optimal |              9.4             12.6             67.8             90.4            901.7
forest6         0.4 infeasible |      29.6 infeas       0.9 numerr      25.9 numerr      27.9 infeas     402.1 infeas
galenet         0.0 infeasible |       0.5 infeas       0.4 numerr      25.6 numerr       0.1 infeas       2.7 infeas
greenbea      140.8    optimal |    20379.7 limit     402.6 numerr     475.2 numerr    30001.4 limit    30001.4 limit
gt2             0.6    optimal |              0.6              0.5             12.3              0.4              5.2
israel          2.1    optimal |              2.9              4.3             28.6             25.4            308.4
klein1          1.2 infeasible |    11375.4 limit      12.0 numerr     202.8 numerr    11329.1 limit    30001.6 limit
perold         36.7    optimal |             61.0             64.9            162.7    30000.5 limit    30002.4 limit
refinery        2.6 infeasible |     445.4 infeas       9.5 numerr     100.0 numerr     432.6 infeas    4702.8 infeas
scrs8           5.4    optimal |              7.1              7.0             58.2            606.0           4088.5
sctest          0.1    optimal |              0.1              0.2              9.6              0.5              9.7
shell           6.9    optimal |             17.8             17.9            140.7            127.5            515.3
stair           8.3    optimal |              6.7              6.9             45.1            602.0           4379.4
standata        2.5    optimal |              4.1              4.3             38.0             15.3             99.0
standgub        2.5    optimal |              4.4              4.5             36.7             16.2             74.9
standmps        2.9    optimal |              7.6              8.2             55.4             29.0            190.9
vol1            5.6 infeasible |    20007.3 limit       7.1 numerr      77.5 numerr    20337.0 limit    30001.7 limit

solved correctly (optimal within 1e-4 rel., or infeasible), out of 19 feasible + 6 infeasible:
  auto_cpu: 21, ipm_cpu: 18, ipm_cuda: 18, pdlp_cpu: 20, pdlp_cuda: 20

relative objective error vs HiGHS (optimal runs only):
model      |         auto_cpu          ipm_cpu         ipm_cuda         pdlp_cpu        pdlp_cuda
25fv47     |          4.4e-08          2.2e-10          2.2e-10          1.6e-06          9.7e-07
80bau3b    |          6.3e-07          3.9e-09          3.9e-09          1.5e-06          1.3e-07
adlittle   |          1.5e-07          7.5e-10          7.5e-10          1.5e-06          1.2e-06
afiro      |          1.0e-06          5.1e-09          5.1e-09          2.0e-07          2.0e-07
avgas      |          4.6e-07          2.3e-09          2.3e-09          9.5e-07          9.5e-07
blending   |          1.4e-08          6.2e-11          6.2e-11          0.0e+00          0.0e+00
e226       |          6.4e-07          3.2e-09          3.2e-09          1.7e-07          1.6e-07
etamacro   |          1.3e-07          3.8e-09          3.8e-09          1.7e-08          4.9e-08
greenbea   |                -                -                -                -                -
gt2        |          3.1e-07          1.6e-09          1.6e-09          0.0e+00          0.0e+00
israel     |          1.2e-06          1.1e-10          1.1e-10          3.6e-07          3.6e-07
perold     |          1.2e-06          0.0e+00          0.0e+00                -                -
scrs8      |          1.4e-08          1.4e-08          1.4e-08          6.6e-08          1.0e-07
sctest     |          1.5e-11          0.0e+00          0.0e+00          7.3e-08          7.3e-08
shell      |          1.2e-08          1.2e-08          1.7e-10          4.4e-07          2.2e-09
stair      |          9.0e-08          4.4e-10          4.4e-10          2.5e-09          7.0e-07
standata   |          2.0e-08          7.9e-11          7.9e-11          1.1e-09          9.0e-09
standgub   |          1.2e-09          1.2e-09          1.2e-09          1.1e-09          1.5e-09
standmps   |          1.4e-06          0.0e+00          0.0e+00          3.7e-08          6.5e-08
```

- Interior point: 18 of the 19 feasible models, to about 1e-9 relative
  objective error, in 0.1 ms to 0.4 s, with the same iteration counts on the
  CPU and the GPU (the GPU is slower here: the models are small and cuDSS'
  analysis and launch overhead dominate). It does not detect infeasibility.
- PDLP: 17 of 19 (perold and greenbea hit the 30 s limit) and 3 of the 6
  infeasible models (forest6, galenet, refinery); the other three
  (bgetam, klein1, vol1) are not certified within the limit. It needs 10-1000x
  more time than the interior-point method on these small, ill-conditioned
  models.
- Auto solves the same 18 feasible models as the interior-point method and
  reports the same 3 infeasible ones through PDLP.
- `greenbea` is unsolved by every method (optimal variables near 3e8).
  Earlier versions reported it as optimal with a 0.13% objective error at
  a 1e-6 tolerance, because the residual tolerances are relative to ||c||;
  the `LpSolution::error_bound` check now rejects such answers.

## Mittelmann LPopt/LPfeas subset (auto, CPU, 300 s limit, eps 1e-6)

16 of the smaller instances from plato.asu.edu/ftp/lptestset (integrality relaxed).
Raw output: `build/mittelmann/auto.log`. Not run: the large instances (>9 MB compressed) and the fome/pds/nug/network/rail directories.

| instance | time (s) | status | objective |
|---|---|---|---|
| brazil3 | 0.6 | optimal | 2.0 |
| chromaticindex1024-7 | 0.7 | optimal | 3.0 |
| datt256_lp | 4.4 | optimal | 256 |
| ex10 | 83.0 | optimal | 100 |
| graph40-40 | 3.8 | optimal | -300 |
| irish-electricity | 300 | time limit | (primal infeasible 0.88) |
| neos-5052403-cygnet | 14.1 | optimal | 179.50043 |
| neos-5251015 | 5.5 | optimal | 0.1058 |
| physiciansched3-3 | 217.6 | optimal | 2.4326948e6 |
| qap15 | 12.0 | optimal | 1040.995 |
| rmine15 | 305 | time limit | gap 6e-4 |
| s100 | 300 | time limit | gap 0.49 |
| s250r10 | 277 | reported optimal, but residuals pres 7e-3 / gap 2e-2: a dual-simplex bug (overflowing steepest-edge weights made row selection skip violated rows), fixed; now time limit | -0.17269 |
| savsched1 | 19.6 | optimal | 217.40357 |
| supportcase10 | 95.4 | optimal | 3.3839 |
| woodlands09 | 15.3 | optimal | ~0 |

## Mini benchmarks after the weak-spot work (CPU, 8 threads, on battery, eps 1e-8 unless noted)

Simplex (HiGHS dual simplex with presolve, 1 thread, for comparison), milliseconds:

```
model        ours   HiGHS      | instance     ours (Auto)   HiGHS
25fv47        108      91      | transport100      27         27
80bau3b       184      77      | transport300     300        176
perold         87      37      | transport700    1030       1419
greenbea      820     147      | mcf10k          1200        384
                               | mcf50k         23400       4830
```

- Dual simplex: 3-4x fewer iterations than before (dual steepest-edge weights
  were collapsing), 9x faster per iteration on large sparse models (hypersparse
  solves): savsched1 39k iterations in 20 s (HiGHS: 34k), neos-5251015 solved in
  5.7 s (HiGHS did not finish in 60 s).
- Interior point: the normal-equations path and the sibling supernode merge take
  transport700 from 14 s to 1.1 s; mcf10k (fill-heavy) is 21 s, so Auto lets the
  simplex go first there (1.0 s).
- PDLP at 1e-6, iterations, adaptive+averaging vs reflected Halpern: 25fv47
  78k -> 57k, 80bau3b 264k -> 129k, e226 36k -> 19k, israel 6.3k -> 4.3k, mcf10k
  2.8k -> 2.2k, transport300 1.9k -> 3.5k, packing20k 9.2k -> 10.7k.
- Not solved within 60 s by us or HiGHS: chromaticindex1024-7, qap15, graph40-40,
  datt256_lp, supportcase10, physiciansched3-3, woodlands09 (simplex);
  the IPM path solves several of them (see the Mittelmann table).
- Known gaps: greenbea (5.6x HiGHS), mcf50k (about 5x), crossover from an
  interior point is not faster than a cold simplex, no Forrest-Tomlin update.

## GPU: PDLP on CUDA vs HiGHS 1.15.1 built with cuPDLP-C (RTX 4060 Laptop)

HiGHS built from source with `-DCUPDLP_GPU=ON`, `--solver pdlp --presolve on
--run_crossover off` (its accuracy option had no visible effect: identical
iteration counts at 1e-4 and 1e-6). Ours: `run_mps --method pdlp` (Halpern), times in ms.

```
instance      ours cuda 1e-4   ours cuda 1e-6   ours cpu 1e-6   HiGHS cuPDLP-C
transport300         408              387            353              630
transport700         818              957           6881             1780
mcf10k               272              302            276              750
mcf50k               401              931           4148             1910
packing20k           310              953           1183             2670
packing200k         1247             5130          47391            19250
```

## Dense factorization kernel (CPU multifrontal)

`dense::syrk_d` is now a packed, register-blocked GEMM-style kernel (AVX-512
vectors, 24x8 tile, KC 256, column blocks as independent OpenMP tasks) and the
panel step computes L21 in parallel row chunks (64-wide panels, 128 for fronts of
3000+). Whole partial factorization of a dense front, GFLOPS (8 cores):

```
front (fs x ns)     before    after
1 thread            ~27..47    49
1000 x 1000          123       126
2000 x 2000          190       212..227
4000 x 4000          116       255..269
```

End to end (IPM, CPU, eps 1e-8): mcf10k 21 s -> 13.7 s, ex10 83 s -> 51 s (about
310 GFLOPS sustained, near this machine's peak), supportcase10 95 s -> 67 s.
Update matrices are pooled between supernodes. cuDSS on the GPU takes 7.3 s (mcf10k),
47 s (ex10), 48 s (supportcase10).

## PDLP polishing (`SolverOptions::pdlp_polish`, off by default)

PDLP to 1e-4, then a primal feasibility solve (objective dropped) and a dual feasibility
solve (b and bounds zeroed), each from the current point. Mixed results at a 1e-6
to 1e-8 target: 80bau3b 5.5 s -> 2.5 s, shell 117 -> 80 ms, but qap15 0.56 -> 1.2 s,
stair 0.37 -> 1.3 s, e226 and 25fv47 slower; perold and greenbea still hit the limit.

## Mittelmann subset, second run (concurrent Auto, 120 s limit, eps 1e-6)

After: concurrent Auto (simplex thread + IPM/PDLP thread), Markowitz basis LU, active-set
pivot row, time-adaptive refactorization, AMD supervariables, the extended presolve and the
faster dense kernel. Seconds; the earlier sequential run (300 s limit) is in the first table.

```
instance               CPU       CUDA          instance               CPU       CUDA
brazil3                0.4        0.8          physiciansched3-3     51.0       24.9
chromaticindex1024-7   1.8        2.2          qap15                 12.5        3.8
datt256_lp            20.0        5.6          rmine15               93.8   limit
ex10                   1.7       41.5          s100               limit      limit
graph40-40             9.5        6.5          s250r10               81.4       16.5
irish-electricity      ~19  (note)               savsched1            53.1       31.8
neos-5052403-cygnet   36.3       12.4          supportcase10         37.8       17.8
neos-5251015          12.4        6.7          woodlands09           22.7       14.1
```

15 of 16 solve on the CPU and 14 of 16 on CUDA (first run: 12 of 16 with 300 s).
Objectives agree with the first run to 1e-6 or better where both solved.

ex10 is an outlier: 1.7 s on the CPU but 41.5 s on CUDA. Not yet investigated. A likely
cause (unverified): the simplex thread finishes early on CUDA too, but `solve_lp` joins the
device thread, and a cuDSS analysis/factorization cannot be interrupted (only the CPU
factorization polls the stop flag), so the run waits for it.

Note on irish-electricity (8.3 s CPU / 7.2 s CUDA before the verification below existed,
about 19 s on the CPU with it): this instance has a chain of about 45 constraints that doubles a tiny slack at each
step. The full presolve fixes the chain exactly (forcing rows) and the reduced problem
solves to 2.54626e6, which HiGHS' simplex confirms on the original problem
(2.54625456e6, 100 s) and HiGHS' IPM certifies on our reduced problem. Without those
reductions an interior-point run converges to 2.4544e6, a point feasible only to 3e-4 that
the chain turns into a 3.7% lower objective: an ill-conditioning artifact, not a better
optimum. The postsolved duals of the full presolve are garbage (residual 38; interior-point
round-off amplified by the chain), so `solve_lp` now (1) verifies the tolerances on the
ORIGINAL problem after postsolve, (2) on failure retries with the conservative reductions
for at most max(10 s, time used), and (3) if that does not verify either, returns the first
result with `LpSolution::duals_verified = false` (status stays Optimal when the primal point
passes; if even the primal fails the status is NumericalError). My first version of the
check retried without a time bound and discarded the correct result; the bound and the flag
are the fix.

## Mittelmann subset, third run (PDLP slice first on the device thread)

Seconds, previous run -> this run (`*` = time limit, 120 s). Changes: the device thread of
concurrent Auto starts with a PDLP slice (2-8 s) on problems with 50000+ nonzeros and before an
expensive interior-point factorization; PDLP ray tolerance tied to the requested accuracy.

```
brazil3                CPU     0.4 ->     0.2   CUDA     0.8 ->     0.4
chromaticindex1024-7   CPU     1.8 ->     0.2   CUDA     2.2 ->     0.4
datt256_lp             CPU    20.0 ->     1.3   CUDA     5.6 ->     0.7
ex10                   CPU     1.7 ->     0.7   CUDA    41.5 ->     0.6
graph40-40             CPU     9.5 ->     1.5   CUDA     6.5 ->     0.8
irish-electricity      CPU     8.3 ->    29.0   CUDA     7.2 ->    26.5
neos-5052403-cygnet    CPU    36.3 ->    26.2   CUDA    12.4 ->    18.2
neos-5251015           CPU    12.4 ->     8.7   CUDA     6.7 ->     1.8
physiciansched3-3      CPU    51.0 ->    65.3   CUDA    24.9 ->    32.8
qap15                  CPU    12.5 ->     0.7   CUDA     3.8 ->     0.4
rmine15                CPU    93.8 ->  120.3*   CUDA  122.6* ->  122.0*
s100                   CPU  121.3* ->  120.7*   CUDA  120.1* ->  120.1*
s250r10                CPU    81.4 ->    79.4   CUDA    16.5 ->    21.8
savsched1              CPU    53.1 ->     4.0   CUDA    31.8 ->     1.1
supportcase10          CPU    37.8 ->    48.6   CUDA    17.8 ->    26.7
woodlands09            CPU    22.7 ->    16.8   CUDA    14.1 ->     2.9
```

14 of 16 solve on both backends (CPU: rmine15 slipped from 94 s to the limit; CUDA: unchanged).
Gains are large where PDLP is the fast method (qap15 12.5 -> 0.7 s, savsched1 53 -> 4 s and 32 -> 1.1 s,
ex10 CUDA 41.5 -> 0.6 s, datt256_lp 20 -> 1.3 s, graph40-40 9.5 -> 1.5 s). Where the interior-point
method is the winner the PDLP slice is pure overhead: physiciansched3-3 +14 s / +8 s, irish-electricity
+21 s (PDLP slice plus the postsolve-verification retry described above), neos-5052403-cygnet CUDA
+6 s, s250r10 CUDA +5 s, rmine15 CPU past the limit. supportcase10 on CUDA varies between runs (4.5 s
in an isolated run, 26.7 s here), which I have not explained. A better policy would run PDLP and the
interior-point method concurrently on the device instead of in slices; not done.

## Mittelmann subset, fourth run (defaults: PDLP-first slice on for CUDA, off for CPU)

Seconds, third run -> this run (`*` = 120 s limit). Changes since the third run: `pdlp_first` is
now automatic (CUDA on, CPU off), looser presolve aggregation (fill <= 64, <= 12 entries),
parallel-column merging, refactor margin 2.0 for bases over 10000 rows.

```
instance               CPU                    CUDA
brazil3                0.2 ->   0.3           0.4 ->   0.5
chromaticindex1024-7   0.2 ->   2.0           0.4 ->   0.4
datt256_lp             1.3 ->   3.7           0.7 ->   0.8
ex10                   0.7 ->   1.6           0.6 ->   0.6
graph40-40             1.5 ->   8.7           0.8 ->   0.9
irish-electricity     29.0 ->  17.5          26.5 ->  26.5
neos-5052403-cygnet   26.2 ->  17.9          18.2 ->  18.5
neos-5251015           8.7 ->   3.1           1.8 ->   1.8
physiciansched3-3     65.3 ->  50.7          32.8 ->  31.4
qap15                  0.7 ->  11.3           0.4 ->   0.6
rmine15             120.3* ->  90.0         122.0* -> 121.2*
s100                120.7* -> 120.3*        120.1* -> 120.0*
s250r10               79.4 ->  71.8          21.8 ->  21.9
savsched1              4.0 ->  22.9           1.1 ->   1.5
supportcase10         48.6 ->  38.3          26.7 ->   4.7
woodlands09           16.8 ->  20.3           2.9 ->   3.7
```

15 of 16 solve on the CPU (rmine15 is back under the limit: 90 s) and 14 of 16 on CUDA.

CUDA is unchanged, as intended (the slice stays on there); supportcase10's 26.7 -> 4.7 s is run-to-run
variation, not a change I can attribute (it was 4.5 s in an isolated run earlier, 26.7 s in the third table).

On the CPU the slice is off now, so the instances it helped got slower: qap15 0.7 -> 11.3 s,
savsched1 4.0 -> 22.9 s, graph40-40 1.5 -> 8.7 s, datt256_lp 1.3 -> 3.7 s, chromaticindex1024-7 0.2 -> 2.0 s,
ex10 0.7 -> 1.6 s. The instances where the interior-point method wins got faster: irish-electricity
29 -> 17.5 s, neos-5052403-cygnet 26 -> 18 s, neos-5251015 8.7 -> 3.1 s, physiciansched3-3 65 -> 51 s,
supportcase10 49 -> 38 s, rmine15 timeout -> 90 s (presolve and slice effects are not separated).
Net on the CPU: 6 instances got faster and 7 slower; the total over the 16 is lower (about 80 s
gained against 45 s lost, rmine15 counted as 30 s), but it is close, and the slice is what makes
qap15 and savsched1 fast. `--pdlp-first` forces it on.

## MILP: branch and bound vs HiGHS 1.15.1 (CPU, 1 thread, 60 s limit, gap 1e-4)

Small MIPLIB-1 models from the HiGHS repository (`benchmarks/solver/fetch_mip.sh`), ours via
`build/run_milp`, HiGHS via `highspy` (`mip_rel_gap 1e-4`, threads 1). No MIP presolve, no cuts.

```
model      ours ms    nodes  status      objective          HiGHS ms  nodes  objective
bell5       14397   287223   optimal     8966406.492           260     180  8966406.492
dcmulti       494     2512   optimal     188182                898       5  188182
egout        2716    83388   optimal     568.1007               10       1  568.1007
flugpl          3      432   optimal     1201500                78      89  1201500
gams10am      0.1        0   infeasible  -                       0       -  infeasible
gas11         1.8        0   unbounded   -                       6       -  unbounded
gesa2       60000   126687   feasible    26036435 (gap 1.2%)   413       1  25779856
gt2            42     1296   optimal     21166                  32       1  21166
lseu          187     8504   optimal     1120                  157       7  1120
p01           0.1        0   optimal     263                     4       1  263
p0548       60000   515459   feasible    34514 (bound 8392)     51       0  8691
rgn            99     2718   optimal     82.2                  177       1  82.2
```

Objectives agree wherever both finish. Ours is slower almost everywhere it needs real work: HiGHS
closes most of these at the root (presolve, cuts, heuristics) while plain branch and bound needs
thousands of nodes; p0548 and gesa2 are the clearest cases. Our node rate is fine (bell5: 24k nodes/s),
so the next gains are presolve and cuts, not LP speed. Ablation (same models, 30 s): bound propagation cuts
nodes by 10x on flugpl (4767 -> 432) and 7x on gt2 (8024 -> 1177); root diving is roughly neutral on these.

## MILP with MIP presolve (same 12 models, 60 s, deterministic node LPs)

```
model      no presolve (ms, nodes)        MIP presolve (ms, nodes)      HiGHS (ms, nodes)
bell5        10223  258677  optimal         10792  258669  optimal          260   180
dcmulti        551    3121  optimal           404    2346  optimal          898     5
egout         2777   83476  optimal           124    5392  optimal           10     1
flugpl           4     499  optimal             3     446  optimal           78    89
gams10am         0       0  infeasible          0       0  infeasible         0     -
gas11            2       0  unbounded           0       0  unbounded          6     -
gesa2        60000  128297  feasible        60056  128267  feasible         413     1
gt2             55    1800  optimal            56    1686  optimal           32     1
lseu           229    9950  optimal            95    4212  optimal          157     7
p01              0       0  optimal             0       0  optimal            4     1
p0548        60000  634957  feasible        10254  106189  optimal           51     0
rgn            104    3010  optimal           105    3010  optimal          177     1
```

10 of 12 solve to optimality (9 without presolve). The presolve's clearest wins are p0548
(unsolved -> 10 s), egout (22x) and lseu (2.4x). Two things that were NOT presolve effects and cost
time to find: the first version regressed gt2 from 40 ms to a timeout, which turned out to be (1) the
search depending on child order (always-up solves gt2 in 3-5k nodes but breaks bell5; always-nearest
is the reverse; the final rule follows pseudocosts) and (2) run-to-run non-determinism from the
timing-based LP refactorization (identical problem: 1.9k to 350k nodes), fixed by a fixed refactor
cadence for node LPs. Repeated runs now give identical node counts.
Still behind HiGHS wherever cuts decide the search (bell5: 259k nodes against 180; gesa2).

## MILP with root cuts (same 12 models, 60 s, deterministic)

MIP presolve on in every column; cuts = complemented MIR + Gomory mixed-integer cuts from the
tableau, rounds until the bound stalls, inactive cuts purged, incumbents verified on the uncut model.

```
model      presolve only              + MIR cuts                 + MIR + Gomory + purge      HiGHS
           ms      nodes              ms      nodes              ms      nodes   status      ms    nodes
bell5     10282   258669             10364   258669              2687    61519   optimal     260     180
dcmulti     404     2346               414     2346               353      290   optimal     898       5
egout       125     5392                26      954                14      159   optimal      10       1
flugpl        3      446                 3      446                 6      280   optimal      78      89
gesa2     61687   116203             61899   114872             56948   111292   optimal     413       1
gt2          85     1686               165     2728                88     1191   optimal      32       1
lseu         97     4212               214     4128                74     1040   optimal     157       7
p0548     10699   106189              2397    14095               148      370   optimal      51       0
rgn         108     3010               214     2462               150     2464   optimal     177       1
(gams10am infeasible, gas11 unbounded, p01 trivial: unchanged)
```

All 10 feasible models are now proven optimal within 60 s with HiGHS' objective (gesa2 at 57 s is
the closest call). Steps that mattered: single-row MIR alone helps egout and p0548 but finds nothing
on bell5 or dcmulti; the Gomory cuts (nodes 8x fewer on dcmulti, 34x on egout, 287x on p0548) are what makes
the difference; the first Gomory version made node LPs 30x slower on small models (rgn 108 ms ->
2.4 s) because every cut stayed in the LP, and purging the cuts that are not binding at the root
removed that cost. Still behind HiGHS on the models it solves at the root by presolve and heuristics
(bell5: 62k nodes against 180; gesa2 and p0548 closer).
