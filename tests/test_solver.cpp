// Optimization solver tests: model, MPS I/O, presolve, scaling, PDLP.
//
// CPU build:   make test_solver_cpu
// CUDA build:  make test_solver

#ifdef PANINI_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "sparse/sparse.h"
#include "solver/io/mps.h"
#include "solver/model.h"
#include "solver/presolve/presolve.h"
#include "solver/solver.h"
#include "solver/lp/simplex.h"
#include "test_common.h"

#include <sstream>

using namespace Panini;
using namespace Panini::Solver;

static const int RUNS = 2;

static bool
near(double a, double b, double tol = 1e-9)
{
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::abs(a - b) <= tol * (1 + std::abs(a) + std::abs(b));
}

static bool
near_vec(const std::vector<double> &a, const std::vector<double> &b,
    double tol = 1e-9)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!near(a[i], b[i], tol)) return false;
    return true;
}

// Small helper for building problems by hand.
struct LpBuilder {
    size_t m, n;
    Sparse::CooBuilder<double> coo;
    LpProblem p;
    LpBuilder(size_t rows, size_t cols) : m(rows), n(cols), coo(rows, cols)
    {
        p.c.assign(n, 0);
        p.col_lb.assign(n, 0);
        p.col_ub.assign(n, kInf);
        p.row_lb.assign(m, -kInf);
        p.row_ub.assign(m, kInf);
    }
    LpBuilder &a(size_t i, size_t j, double v) { coo.add(i, j, v); return *this; }
    LpProblem build() { p.A = coo.build(); return p; }
};

// ─── MPS ─────────────────────────────────────────────────────────────────────
static const char *kMpsFull = R"(* comment
NAME          TESTLP
OBJSENSE
    MAX
ROWS
 N  COST
 N  EXTRA
 L  LIM1
 G  LIM2
 E  MYEQN
 G  RNG1
 L  RNG2
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0   EXTRA        5.0
    MARKER                 'MARKER'                 'INTEND'
    X2        COST         2.0   LIM1         1.0
    X2        MYEQN       -1.0   RNG1         1.0
    X3        COST        -1.0   MYEQN        1.0
    X3        RNG2         2.0
    X4        COST         0.5   LIM2         1.0
RHS
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0   COST        -3.0
    RHS       RNG1         2.0   RNG2         9.0
RANGES
    RNG       RNG1         4.0   RNG2         5.0
    RNG       MYEQN       -2.0
BOUNDS
 UP BND       X1           4.0
 LO BND       X2          -1.0
 UP BND       X2           1.0
 MI BND       X3
 UP BND       X3           5.0
 FR BND       X4
ENDATA
)";

static void
test_mps_read()
{
    std::istringstream in(kMpsFull);
    LpProblem p = read_mps(in);
    bool ok = p.name == "TESTLP" && p.rows() == 5 && p.cols() == 4 &&
              p.maximize;
    tlog("SOLVER_MPS", "read dimensions/name/sense", ok, 0, 0);
    // maximize: stored c is negated
    tlog("SOLVER_MPS", "objective (negated for MAX) and offset",
        near_vec(p.c, {-1, -2, 1, -0.5}) && near(p.offset, -3.0), 0, 0);
    // rows: LIM1 (L,4), LIM2 (G,1), MYEQN (E 7, range -2 -> [5,7]),
    // RNG1 (G 2, +4 -> [2,6]), RNG2 (L 9, 5 -> [4,9])
    tlog("SOLVER_MPS", "row bounds incl. RANGES",
        near_vec(p.row_lb, {-kInf, 1, 5, 2, 4}) &&
            near_vec(p.row_ub, {4, kInf, 7, 6, 9}),
        0, 0);
    // X1 int [0,4], X2 [-1,1], X3 (-inf,5], X4 free
    tlog("SOLVER_MPS", "column bounds",
        near_vec(p.col_lb, {0, -1, -kInf, -kInf}) &&
            near_vec(p.col_ub, {4, 1, 5, kInf}),
        0, 0);
    tlog("SOLVER_MPS", "integer markers",
        p.is_integer.size() == 4 && p.is_integer[0] == 1 &&
            p.is_integer[1] == 0 && p.has_integers(),
        0, 0);
    // dropped N row EXTRA must not add a row or entry
    Sparse::Csr<double, int32_t> &A = p.A;
    tlog("SOLVER_MPS", "extra N row dropped, matrix entries",
        A.nnz() == 8 && A.validate().empty() && p.validate().empty(), 0, 0);
}

static void
test_mps_edge_cases()
{
    // No set names in RHS/BOUNDS, negative UP with default lower bound.
    const char *txt = R"(NAME T
ROWS
 N obj
 G r1
COLUMNS
 x obj 1 r1 1
 y obj 1 r1 1
RHS
 r1 3
BOUNDS
 UP x -2
 BV y
ENDATA
)";
    std::istringstream in(txt);
    LpProblem p = read_mps(in);
    tlog("SOLVER_MPS", "no set names; UP<0 sets lb=-inf; BV",
        near_vec(p.row_lb, {3}) && near_vec(p.col_lb, {-kInf, 0}) &&
            near_vec(p.col_ub, {-2, 1}) && p.is_integer.size() == 2 &&
            p.is_integer[1] == 1,
        0, 0);

    auto throws = [](const char *s) {
        try {
            std::istringstream is(s);
            read_mps(is);
        } catch (const std::exception &) { return true; }
        return false;
    };
    tlog("SOLVER_MPS", "unknown row name is an error",
        throws("NAME T\nROWS\n N o\n L r\nCOLUMNS\n x nosuch 1\nENDATA\n"), 0, 0);
    tlog("SOLVER_MPS", "bad number is an error",
        throws("NAME T\nROWS\n N o\nCOLUMNS\n x o 1.2.3\nENDATA\n"), 0, 0);
    tlog("SOLVER_MPS", "unknown section is an error",
        throws("NAME T\nBOGUS\nENDATA\n"), 0, 0);
    tlog("SOLVER_MPS", "semi-continuous bound rejected",
        throws("NAME T\nROWS\n N o\nCOLUMNS\n x o 1\nBOUNDS\n SC B x 5\nENDATA\n"),
        0, 0);
}

static void
test_mps_roundtrip()
{
    std::istringstream in(kMpsFull);
    LpProblem p = read_mps(in);
    std::ostringstream out;
    write_mps(out, p);
    std::istringstream in2(out.str());
    LpProblem q = read_mps(in2);
    bool same = q.rows() == p.rows() && q.cols() == p.cols() &&
                q.maximize == p.maximize && near_vec(q.c, p.c, 1e-14) &&
                near_vec(q.row_lb, p.row_lb, 1e-14) &&
                near_vec(q.row_ub, p.row_ub, 1e-14) &&
                near_vec(q.col_lb, p.col_lb, 1e-14) &&
                near_vec(q.col_ub, p.col_ub, 1e-14) &&
                near(q.offset, p.offset, 1e-14) &&
                q.is_integer == p.is_integer && q.A.nnz() == p.A.nnz();
    if (same) {
        auto Dp = p.A.to_dense(), Dq = q.A.to_dense();
        for (size_t i = 0; i < Dp.size(); ++i)
            same = same && Dp.data[i] == Dq.data[i];
    }
    tlog("SOLVER_MPS", "write -> read round trip", same, 0, 0);

    // Random problems round-trip exactly (17 significant digits).
    std::mt19937 g(5);
    std::uniform_real_distribution<double> u(-3, 3);
    bool all = true;
    for (int trial = 0; trial < 5; ++trial) {
        LpBuilder b(6, 7);
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 7; ++j)
                if (u(g) > 0.5) b.a(i, j, u(g));
        for (size_t j = 0; j < 7; ++j) {
            b.p.c[j] = u(g);
            b.p.col_lb[j] = (j % 3 == 0) ? -kInf : u(g) - 4;
            b.p.col_ub[j] = (j % 4 == 0)
                                ? kInf
                                : (std::isinf(b.p.col_lb[j])
                                          ? u(g) + 5
                                          : b.p.col_lb[j] + 2 + std::abs(u(g)));
        }
        for (size_t i = 0; i < 6; ++i) {
            double lo = u(g), hi = lo + std::abs(u(g));
            if (i % 3 == 0) { b.p.row_lb[i] = lo; b.p.row_ub[i] = lo; }
            else if (i % 3 == 1) { b.p.row_lb[i] = lo; b.p.row_ub[i] = kInf; }
            else { b.p.row_lb[i] = lo; b.p.row_ub[i] = hi + 0.5; }
        }
        b.p.offset = u(g);
        LpProblem p0 = b.build();
        std::ostringstream o;
        write_mps(o, p0);
        std::istringstream i2(o.str());
        LpProblem q0 = read_mps(i2);
        all = all && near_vec(q0.c, p0.c, 1e-14) &&
              near_vec(q0.row_lb, p0.row_lb, 1e-14) &&
              near_vec(q0.row_ub, p0.row_ub, 1e-14) &&
              near_vec(q0.col_lb, p0.col_lb, 1e-14) &&
              near_vec(q0.col_ub, p0.col_ub, 1e-14) &&
              near(q0.offset, p0.offset, 1e-14) && q0.A.nnz() == p0.A.nnz();
    }
    tlog("SOLVER_MPS", "random problems round trip", all, 0, 0);
}

// ─── model evaluation ────────────────────────────────────────────────────────
static void
test_evaluate()
{
    // min x1 + 2 x2  s.t. x1 + x2 >= 3, x >= 0.  Optimal x=(3,0), y=1.
    LpBuilder b(1, 2);
    b.a(0, 0, 1).a(0, 1, 1);
    b.p.c = {1, 2};
    b.p.row_lb[0] = 3;
    LpProblem p = b.build();
    LpSolution s = evaluate_solution(p, {3, 0}, {1});
    tlog("SOLVER_MODEL", "evaluate_solution at the optimum",
        near(s.primal_objective, 3) && near(s.dual_objective, 3) &&
            s.primal_residual == 0 && s.dual_residual == 0 && s.gap == 0 &&
            near_vec(s.z, {0, 1}),
        0, 0);
    LpSolution bad = evaluate_solution(p, {1, 0}, {-1}); // infeasible, y<0 on -inf ub
    tlog("SOLVER_MODEL", "evaluate_solution flags violations",
        near(bad.primal_residual, 2) && bad.dual_residual > 0, 0, 0);
    // error_bound: ~0 at the optimum, large when a residual meets big variables
    tlog("SOLVER_MODEL", "error_bound is tight at the optimum", s.error_bound < 1e-12, 0, 0);
    LpSolution off = evaluate_solution(p, {1e6, 0}, {1});
    tlog("SOLVER_MODEL", "error_bound reflects the size of the variables",
        off.error_bound > 0.5 * off.gap && off.error_bound >= off.gap, 0, 0);
    LpProblem q = p;
    q.col_lb[0] = 5;
    q.col_ub[0] = 4;
    tlog("SOLVER_MODEL", "validate rejects lb > ub", !q.validate().empty(), 0, 0);
}

// ─── presolve ────────────────────────────────────────────────────────────────
static void
test_presolve_singleton()
{
    // min 3 x1 + 2 x2  s.t.  r1: x1 >= 1 (singleton), r2: x1 + x2 >= 3, x >= 0
    // optimum x = (1, 2), duals y = (1, 2), z = (0, 0), objective 7.
    LpBuilder b(2, 2);
    b.a(0, 0, 1).a(1, 0, 1).a(1, 1, 1);
    b.p.c = {3, 2};
    b.p.row_lb = {1, 3};
    LpProblem p = b.build();
    Presolve pre(p);
    bool ok = pre.status() == Status::NotSolved && pre.removed_rows() == 1 &&
              pre.reduced().rows() == 1 && near(pre.reduced().col_lb[0], 1);
    tlog("SOLVER_PRESOLVE", "singleton row -> column bound", ok, 0, 0);
    LpSolution r;
    r.status = Status::Optimal;
    r.x = {1, 2};
    r.y = {2}; // reduced problem: row r2 dual
    LpSolution s = pre.postsolve(r);
    tlog("SOLVER_PRESOLVE", "singleton row dual recovered",
        near_vec(s.y, {1, 2}) && near_vec(s.z, {0, 0}) &&
            near(s.primal_objective, 7) && near(s.dual_objective, 7) &&
            s.primal_residual < 1e-12 && s.dual_residual < 1e-12,
        0, 0);

    // Same problem where the singleton bound is NOT active: min x1 + 2 x2.
    LpBuilder b2(2, 2);
    b2.a(0, 0, 1).a(1, 0, 1).a(1, 1, 1);
    b2.p.c = {1, 2};
    b2.p.row_lb = {1, 3};
    LpProblem p2 = b2.build();
    Presolve pre2(p2);
    LpSolution r2;
    r2.status = Status::Optimal;
    r2.x = {3, 0};
    r2.y = {1};
    LpSolution s2 = pre2.postsolve(r2);
    tlog("SOLVER_PRESOLVE", "inactive singleton row keeps y=0",
        near_vec(s2.y, {0, 1}) && near(s2.primal_objective, 3) &&
            near(s2.dual_objective, 3),
        0, 0);

    // Negative coefficient and an upper bound: -2 x1 <= -6  (x1 >= 3),
    // min x1 + x2, x1 + x2 >= 1: optimum x=(3,0), y1 = ?
    // z1 = 1 - (-2) y1 - y2 ; y2 = 0 (row inactive), z1 must be 0 at
    // the tightened bound with dual: y1 = z_red/a = 1/(-2) = -0.5.
    LpBuilder b3(2, 2);
    b3.a(0, 0, -2).a(1, 0, 1).a(1, 1, 1);
    b3.p.c = {1, 1};
    b3.p.row_ub[0] = -6;
    b3.p.row_lb[1] = 1;
    LpProblem p3 = b3.build();
    PresolveOptions keep_cols; keep_cols.parallel_cols = false; // the test feeds postsolve a reduced solution of fixed shape
    Presolve pre3(p3, keep_cols);
    LpSolution r3;
    r3.status = Status::Optimal;
    r3.x = {3, 0};
    r3.y = {0};
    LpSolution s3 = pre3.postsolve(r3);
    tlog("SOLVER_PRESOLVE", "negative-coefficient singleton row",
        pre3.removed_rows() == 1 && near_vec(s3.y, {-0.5, 0}) &&
            near_vec(s3.z, {0, 1}) && s3.primal_residual < 1e-12 &&
            s3.dual_residual < 1e-12 && near(s3.primal_objective, s3.dual_objective),
        0, 0);
}

static void
test_presolve_other()
{
    // Fixed column, empty column, empty row.
    // min x1 + 2 x2 + 3 x3 + 4 x4   s.t.
    //   r1: x1 + x2 >= 5
    //   r2: (empty) in [-1, 1]
    // x1 fixed at 2 -> r1: x2 >= 3.  x3 empty col, cost 3>0 -> at lb 0.
    // x4 empty col with ub 6 and cost -4 < 0 -> at ub.
    LpBuilder b(2, 4);
    b.a(0, 0, 1).a(0, 1, 1);
    b.p.c = {1, 2, 3, -4};
    b.p.col_lb = {2, 0, 0, 0};
    b.p.col_ub = {2, kInf, kInf, 6};
    b.p.row_lb = {5, -1};
    b.p.row_ub = {kInf, 1};
    LpProblem p = b.build();
    Presolve pre(p);
    // Everything reduces away except x2 (which becomes a singleton bound and
    // then an empty column at its lower bound 3).
    bool ok = pre.status() == Status::NotSolved && pre.reduced().rows() == 0 &&
              pre.reduced().cols() == 0;
    tlog("SOLVER_PRESOLVE", "fixed/empty/singleton reduce to nothing", ok, 0, 0);
    LpSolution s = pre.postsolve(LpSolution{});
    // x = (2, 3, 0, 6): objective 2 + 6 + 0 - 24 = -16. Duals: r1 gets
    // z2 = 2 - y1 = 0 -> y1 = 2 (singleton takes it), r2 y=0.
    tlog("SOLVER_PRESOLVE", "postsolve of a fully reduced problem",
        near_vec(s.x, {2, 3, 0, 6}) && near_vec(s.y, {2, 0}) &&
            near(s.primal_objective, -16) &&
            near(s.dual_objective, s.primal_objective) &&
            s.primal_residual < 1e-12 && s.dual_residual < 1e-12,
        0, 0);

    // Infeasible: empty row that requires >= 1.
    LpBuilder bi(2, 1);
    bi.a(0, 0, 1);
    bi.p.row_lb = {0, 1};
    bi.p.row_ub = {kInf, kInf};
    LpProblem pi_ = bi.build();
    tlog("SOLVER_PRESOLVE", "empty infeasible row detected",
        Presolve(pi_).status() == Status::Infeasible, 0, 0);

    // Infeasible: conflicting singleton rows x >= 3 and x <= 1.
    LpBuilder bc(2, 1);
    bc.a(0, 0, 1).a(1, 0, 1);
    bc.p.row_lb = {3, -kInf};
    bc.p.row_ub = {kInf, 1};
    LpProblem pc = bc.build();
    tlog("SOLVER_PRESOLVE", "conflicting singleton rows -> infeasible",
        Presolve(pc).status() == Status::Infeasible, 0, 0);

    // Unbounded: empty column with negative cost and no upper bound.
    LpBuilder bu(1, 2);
    bu.a(0, 0, 1);
    bu.p.c = {1, -1};
    bu.p.row_lb = {0};
    LpProblem pu = bu.build();
    tlog("SOLVER_PRESOLVE", "empty column unbounded direction",
        Presolve(pu).status() == Status::Unbounded, 0, 0);

    // Options turn reductions off.
    PresolveOptions off;
    off.empty_rows = off.empty_cols = off.fixed_cols = off.singleton_rows =
        off.duplicate_rows = off.activity_rows = off.dual_fixing = off.singleton_cols =
            off.doubleton_equations = off.aggregate = off.parallel_cols = false;
    Presolve none(p, off);
    tlog("SOLVER_PRESOLVE", "all reductions disabled leaves problem intact",
        none.reduced().rows() == 2 && none.reduced().cols() == 4, 0, 0);
}

// Random LPs rich in the structures the presolve rules target: every rule's
// postsolve must give the same objective as a solve without presolve, with
// primal and dual residuals at solver accuracy on the original problem.
static void
test_presolve_random()
{
    std::mt19937 g(77);
    std::uniform_real_distribution<double> u(-1, 1), pos(0.5, 2);
    int ok_count = 0, total = 0;
    double worst = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const int m0 = 12 + trial % 7, n0 = 18 + trial % 5;
        const int extra_rows = 8, extra_cols = 8;
        LpBuilder b(m0 + extra_rows, n0 + extra_cols);
        // a feasible core: rows at a known point x0 >= 0, with slack either side
        std::vector<double> x0(n0 + extra_cols);
        for (auto &v : x0) v = pos(g);
        for (int i = 0; i < m0; ++i) {
            double act = 0;
            for (int j = 0; j < n0; ++j)
                if (u(g) > 0.55) { const double a = u(g); b.a(i, j, a); act += a * x0[j]; }
            const int kind = static_cast<int>(g() % 3);
            if (kind == 0) { b.p.row_lb[i] = b.p.row_ub[i] = act; }
            else if (kind == 1) { b.p.row_lb[i] = act - pos(g); }
            else { b.p.row_ub[i] = act + pos(g); }
        }
        for (int j = 0; j < n0; ++j) { b.p.c[j] = u(g) + 0.3; b.p.col_ub[j] = x0[j] + 3; }
        int r = m0, c = n0;
        // doubleton equations a x_j + b x_k = rhs (x_k a new column), incl. x_j + x_k = 0
        for (int t = 0; t < 3; ++t, ++r, ++c) {
            const int j = static_cast<int>(g() % n0);
            const double a = u(g) >= 0 ? 1.0 : pos(g), bb = pos(g);
            b.a(r, j, a).a(r, c, bb);
            if (t == 0) { b.p.row_lb[r] = b.p.row_ub[r] = 0; b.p.col_ub[c] = kInf; } // forces x_j = x_c = 0 when a > 0
            else { b.p.row_lb[r] = b.p.row_ub[r] = a * x0[j] + bb * x0[c]; b.p.col_ub[c] = x0[c] + 1; }
            b.p.c[c] = u(g);
            // the new column also appears in a core row
            b.a(static_cast<int>(g() % m0), c, u(g));
        }
        // implied-free column singleton in an equality
        {
            const int j1 = static_cast<int>(g() % n0), j2 = static_cast<int>(g() % n0);
            b.a(r, j1, 1).a(r, j2 == j1 ? (j1 + 1) % n0 : j2, 1).a(r, c, 1);
            b.p.row_lb[r] = b.p.row_ub[r] = 100; // x_c = 100 - x_j1 - x_j2 >= 90 > 0
            b.p.col_lb[c] = -kInf; b.p.col_ub[c] = kInf; b.p.c[c] = u(g);
            ++r; ++c;
        }
        // free zero-cost singleton in an inequality
        {
            const int j1 = static_cast<int>(g() % n0);
            b.a(r, j1, 1).a(r, c, 2);
            b.p.row_lb[r] = -kInf; b.p.row_ub[r] = 1;
            b.p.col_lb[c] = -kInf; b.p.col_ub[c] = kInf; b.p.c[c] = 0;
            ++r; ++c;
        }
        // forcing row: sum of boxed columns at their lower bounds equals the upper bound
        {
            double lo = 0;
            for (int t = 0; t < 3; ++t, ++c) {
                b.a(r, c, 1); b.p.col_lb[c] = 1; b.p.col_ub[c] = 2; lo += 1; b.p.c[c] = u(g);
                b.a(static_cast<int>(g() % m0), c, u(g));
            }
            b.p.row_ub[r] = lo;
            ++r;
        }
        // redundant row
        {
            const int j1 = static_cast<int>(g() % n0);
            b.a(r, j1, 1);
            b.a(r, (j1 + 3) % n0, 1);
            b.p.row_ub[r] = 1e4;
            ++r;
        }
        for (; c < n0 + extra_cols; ++c) { b.p.c[c] = 1; b.p.col_ub[c] = 1; } // unused columns
        for (; r < m0 + extra_rows; ++r) {}                                       // empty rows
        LpProblem p = b.build();
        SolverOptions on, off;
        on.method = off.method = LpMethod::Simplex;
        on.set_tolerance(1e-9);
        off.set_tolerance(1e-9);
        off.presolve = false;
        LpSolution a = solve_lp<Cpu::HostStorage>(p, on), ref = solve_lp<Cpu::HostStorage>(p, off);
        if (ref.status != Status::Optimal) continue; // the random core can be infeasible
        ++total;
        const double e = std::abs(a.primal_objective - ref.primal_objective) / (1 + std::abs(ref.primal_objective));
        const double res = std::max(a.primal_residual, a.dual_residual);
        worst = std::max({worst, e, res});
        if (a.status == Status::Optimal && e < 1e-7 && res < 1e-7) ++ok_count;
    }
    tlog("SOLVER_PRESOLVE", ("random LPs: presolve + postsolve == no presolve (" +
                                std::to_string(ok_count) + "/" + std::to_string(total) + ")").c_str(),
        total >= 15 && ok_count == total, worst, 0);
}

// Parallel columns: proportional entries and costs are merged, postsolve splits
// the merged variable inside both columns' bounds (also for a negative factor).
static void
test_presolve_parallel()
{
    for (double alpha : {2.0, -2.0}) {
        // min x1 + alpha x2 + 3 x3   (cost of x2 = alpha * cost of x1)
        // s.t. r1: x1 + alpha x2 + x3 >= 4 ; r2: x1 + alpha x2 <= 10
        // x1 in [0,1], x2 in [0,3] (w = x1 + alpha x2), x3 >= 0 at cost 3.
        LpBuilder b(2, 3);
        b.a(0, 0, 1).a(0, 1, alpha).a(0, 2, 1).a(1, 0, 1).a(1, 1, alpha);
        b.p.c = {1, alpha * 1, 3};
        b.p.col_ub = {1, 3, kInf};
        b.p.row_lb = {4, -kInf};
        b.p.row_ub = {kInf, 10};
        LpProblem p = b.build();
        Presolve pre(p);
        SolverOptions o;
        o.method = LpMethod::Simplex;
        o.set_tolerance(1e-9);
        LpSolution s = solve_lp<Cpu::HostStorage>(p, o);
        LpSolution ref;
        {
            SolverOptions off = o;
            off.presolve = false;
            ref = solve_lp<Cpu::HostStorage>(p, off);
        }
        const bool in_bounds = s.x.size() == 3 && s.x[0] >= -1e-9 && s.x[0] <= 1 + 1e-9 &&
                               s.x[1] >= -1e-9 && s.x[1] <= 3 + 1e-9 && s.x[2] >= -1e-9;
        tlog("SOLVER_PRESOLVE", (std::string("parallel columns merged and split (factor ") +
                                    (alpha > 0 ? "+2" : "-2") + ")").c_str(),
            pre.removed_cols() >= 1 && s.status == Status::Optimal &&
                ref.status == Status::Optimal && in_bounds &&
                std::abs(s.primal_objective - ref.primal_objective) < 1e-8 &&
                s.primal_residual < 1e-9 && s.dual_residual < 1e-9,
            std::abs(s.primal_objective - ref.primal_objective), 0);
    }
}

static void
test_presolve_duplicates()
{
    // r1: x1 + x2 >= 2, r2: 2 x1 + 2 x2 >= 6 (implied >= 3); min x1 + 2 x2.
    // optimum x=(3,0): merged row binds via r2: y = (0, 0.5).
    LpBuilder b(2, 2);
    b.a(0, 0, 1).a(0, 1, 1).a(1, 0, 2).a(1, 1, 2);
    b.p.c = {1, 2};
    b.p.row_lb = {2, 6};
    LpProblem p = b.build();
    Presolve pre(p);
    tlog("SOLVER_PRESOLVE", "duplicate rows merged",
        pre.reduced().rows() == 1 && near(pre.reduced().row_lb[0], 3), 0, 0);
    LpSolution r;
    r.status = Status::Optimal;
    r.x = {3, 0};
    r.y = {1};
    LpSolution s = pre.postsolve(r);
    tlog("SOLVER_PRESOLVE", "duplicate row dual goes to the binding row",
        near_vec(s.y, {0, 0.5}) && near(s.primal_objective, 3) &&
            near(s.dual_objective, 3) && s.dual_residual < 1e-12,
        0, 0);

    // Negative multiple with opposite bounds: r1: x1+x2 <= 4, r2: -x1-x2 <= -1
    // (i.e. x1+x2 >= 1).  Merged [1, 4].  min -x1 - x2 -> x1+x2 = 4 (upper
    // of r1).  y1 = -1 (at upper), y2 = 0.
    LpBuilder b2(2, 2);
    b2.a(0, 0, 1).a(0, 1, 1).a(1, 0, -1).a(1, 1, -1);
    b2.p.c = {-1, -1};
    b2.p.row_ub = {4, -1};
    LpProblem p2 = b2.build();
    PresolveOptions keep_cols; keep_cols.parallel_cols = false; // see above: fixed-shape hand-fed solution
    Presolve pre2(p2, keep_cols);
    LpSolution r2;
    r2.status = Status::Optimal;
    r2.x = {4, 0};
    r2.y = {-1};
    LpSolution s2 = pre2.postsolve(r2);
    tlog("SOLVER_PRESOLVE", "duplicate rows with a negative multiple",
        pre2.reduced().rows() == 1 && near_vec(s2.y, {-1, 0}) &&
            near(s2.primal_objective, -4) && near(s2.dual_objective, -4),
        0, 0);

    // Other row binds through the negative multiple: r1: x1+x2 <= 5,
    // r2: -x1-x2 <= -1 wait that gives lb 1; make min x1+x2 -> optimum at
    // lower bound 1 supplied by r2 (its upper bound -1 => a_i x >= 1).
    LpBuilder b3(2, 2);
    b3.a(0, 0, 1).a(0, 1, 1).a(1, 0, -1).a(1, 1, -1);
    b3.p.c = {1, 1};
    b3.p.row_ub = {5, -1};
    LpProblem p3 = b3.build();
    Presolve pre3(p3);
    LpSolution r3;
    r3.status = Status::Optimal;
    r3.x = {1, 0};
    r3.y = {1};
    LpSolution s3 = pre3.postsolve(r3);
    tlog("SOLVER_PRESOLVE", "duplicate binding via negative multiple",
        near_vec(s3.y, {0, -1}) && near(s3.primal_objective, 1) &&
            near(s3.dual_objective, 1) && s3.dual_residual < 1e-12 &&
            s3.primal_residual < 1e-12,
        0, 0);

    // Non-parallel rows must not be merged.
    LpBuilder b4(2, 2);
    b4.a(0, 0, 1).a(0, 1, 1).a(1, 0, 1).a(1, 1, 2);
    b4.p.row_lb = {1, 1};
    LpProblem p4 = b4.build();
    tlog("SOLVER_PRESOLVE", "non-parallel rows are kept",
        Presolve(p4).reduced().rows() == 2, 0, 0);
}

// ─── LPs with a known optimal primal-dual pair ───────────────────────────────
struct KnownLp {
    LpProblem p;
    std::vector<double> x, y; // optimal pair for p
    double obj = 0;
    size_t dup_rows = 0, single_rows = 0, fixed_cols = 0;
};

// Builds min c^T x + offset s.t. row/col bounds such that (x*, y*) satisfy the
// KKT conditions by construction. `extras` adds duplicate rows, singleton rows
// and fixed columns so presolve has work to do.
static KnownLp
make_known_lp(size_t m, size_t n, double density, unsigned seed, bool extras)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u01(0, 1), u11(-1, 1);
    const size_t nfixed = extras ? std::max<size_t>(1, n / 10) : 0;
    const size_t ntot = n + nfixed;

    // matrix (dense pattern draw so every row/column is non-empty)
    Sparse::CooBuilder<double> coo(m, ntot);
    for (size_t i = 0; i < m; ++i) {
        bool any = false;
        for (size_t j = 0; j < ntot; ++j)
            if (u01(g) < density) { coo.add(i, j, u11(g) * 2 + (u01(g) < 0.5 ? 0.3 : -0.3)); any = true; }
        if (!any) coo.add(i, g() % ntot, 1.0 + u01(g));
    }
    for (size_t j = 0; j < ntot; ++j) coo.add(g() % m, j, 0.5 + u01(g)); // column non-empty
    HostMatrix A = coo.build();

    KnownLp k;
    std::vector<double> x(ntot), z(ntot), lb(ntot), ub(ntot);
    for (size_t j = 0; j < ntot; ++j) {
        if (j >= n) { // fixed column
            lb[j] = ub[j] = x[j] = u11(g) * 2;
            z[j] = u11(g); // any sign is fine for a fixed variable
            continue;
        }
        lb[j] = (u01(g) < 0.15) ? -kInf : -2 * u01(g);
        ub[j] = (u01(g) < 0.25) ? kInf
                                : (std::isinf(lb[j]) ? 3 * u01(g) : lb[j] + 1 + 2 * u01(g));
        int st = static_cast<int>(g() % 3); // 0 lower, 1 upper, 2 interior
        if (st == 0 && std::isinf(lb[j])) st = 2;
        if (st == 1 && std::isinf(ub[j])) st = 2;
        if (st == 0) { x[j] = lb[j]; z[j] = 0.1 + u01(g); }
        else if (st == 1) { x[j] = ub[j]; z[j] = -(0.1 + u01(g)); }
        else {
            double lo = std::isinf(lb[j]) ? -2 : lb[j];
            double hi = std::isinf(ub[j]) ? lo + 4 : ub[j];
            x[j] = lo + (hi - lo) * (0.2 + 0.6 * u01(g));
            z[j] = 0;
        }
    }
    std::vector<double> ax(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (int kk = A.row_ptr()[i]; kk < A.row_ptr()[i + 1]; ++kk)
            ax[i] += A.values()[kk] * x[A.col_ind()[kk]];
    std::vector<double> rl(m), ru(m), y(m);
    for (size_t i = 0; i < m; ++i) {
        int t = static_cast<int>(g() % 4);
        if (t == 0) { rl[i] = ru[i] = ax[i]; y[i] = u11(g); }
        else if (t == 1) {
            rl[i] = ax[i]; ru[i] = (u01(g) < 0.4) ? kInf : ax[i] + 3 * u01(g);
            y[i] = 0.1 + u01(g);
        } else if (t == 2) {
            ru[i] = ax[i]; rl[i] = (u01(g) < 0.4) ? -kInf : ax[i] - 3 * u01(g);
            y[i] = -(0.1 + u01(g));
        } else {
            rl[i] = ax[i] - 0.5 - 2 * u01(g); ru[i] = ax[i] + 0.5 + 2 * u01(g);
            if (u01(g) < 0.3) rl[i] = -kInf;
            y[i] = 0;
        }
    }
    std::vector<double> c(z);
    for (size_t i = 0; i < m; ++i)
        for (int kk = A.row_ptr()[i]; kk < A.row_ptr()[i + 1]; ++kk)
            c[A.col_ind()[kk]] += A.values()[kk] * y[i];

    // Extra rows for presolve: duplicates (any scale) and singletons.
    Sparse::CooBuilder<double> full(m, ntot);
    size_t rows = m;
    std::vector<double> xl = rl, xu = ru, yy = y;
    std::vector<std::vector<std::pair<int, double>>> extra_rows;
    if (extras) {
        for (size_t t = 0; t < std::max<size_t>(2, m / 8); ++t) {
            size_t i = g() % m;
            double alpha = (t % 3 == 0) ? -2.0 : (t % 3 == 1 ? 0.5 : 3.0);
            std::vector<std::pair<int, double>> r;
            for (int kk = A.row_ptr()[i]; kk < A.row_ptr()[i + 1]; ++kk)
                r.push_back({A.col_ind()[kk], alpha * A.values()[kk]});
            extra_rows.push_back(r);
            double lo = alpha > 0 ? alpha * rl[i] : alpha * ru[i];
            double hi = alpha > 0 ? alpha * ru[i] : alpha * rl[i];
            xl.push_back(lo); xu.push_back(hi); yy.push_back(0);
            ++k.dup_rows;
        }
        for (size_t t = 0; t < std::max<size_t>(2, n / 10); ++t) {
            size_t j = g() % n;
            double a = (u01(g) < 0.5) ? 1.5 : -0.8;
            extra_rows.push_back({{static_cast<int>(j), a}});
            double act = a * x[j];
            xl.push_back(act - 1 - u01(g)); xu.push_back(act + 1 + u01(g));
            yy.push_back(0);
            ++k.single_rows;
        }
        rows += extra_rows.size();
    }
    Sparse::CooBuilder<double> B(rows, ntot);
    for (size_t i = 0; i < m; ++i)
        for (int kk = A.row_ptr()[i]; kk < A.row_ptr()[i + 1]; ++kk)
            B.add(i, A.col_ind()[kk], A.values()[kk]);
    for (size_t r = 0; r < extra_rows.size(); ++r)
        for (auto &e : extra_rows[r]) B.add(m + r, e.first, e.second);

    k.fixed_cols = nfixed;
    k.p.A = B.build();
    k.p.c = c;
    k.p.col_lb = lb;
    k.p.col_ub = ub;
    k.p.row_lb = xl;
    k.p.row_ub = xu;
    k.p.offset = u11(g) * 5;
    k.x = x;
    k.y = yy;
    k.obj = k.p.objective(x);
    return k;
}

static bool
solution_ok(const LpProblem &p, const LpSolution &s, double obj_ref, double tol)
{
    if (s.status != Status::Optimal) return false;
    LpSolution e = evaluate_solution(p, s.x, s.y);
    double nb = 0, nc = 0;
    for (size_t i = 0; i < p.rows(); ++i) {
        if (std::isfinite(p.row_lb[i])) nb += p.row_lb[i] * p.row_lb[i];
        if (std::isfinite(p.row_ub[i])) nb += p.row_ub[i] * p.row_ub[i];
    }
    for (double v : p.c) nc += v * v;
    return e.primal_residual <= 5 * tol * (1 + std::sqrt(nb)) &&
           e.dual_residual <= 5 * tol * (1 + std::sqrt(nc)) &&
           std::abs(e.primal_objective - obj_ref) <= 20 * tol * (1 + std::abs(obj_ref)) &&
           e.gap <= 20 * tol * (1 + std::abs(e.primal_objective) + std::abs(e.dual_objective));
}

static void
test_known_generator()
{
    bool ok = true;
    for (unsigned seed = 1; seed <= 6; ++seed) {
        for (bool ex : {false, true}) {
            KnownLp k = make_known_lp(20 + seed * 5, 30 + seed * 4, 0.25, seed, ex);
            LpSolution e = evaluate_solution(k.p, k.x, k.y);
            ok = ok && k.p.validate().empty() && e.primal_residual < 1e-9 &&
                 e.dual_residual < 1e-9 && e.gap < 1e-7 &&
                 std::abs(e.primal_objective - k.obj) < 1e-9;
        }
    }
    tlog("SOLVER_LP", "generator produces exact primal-dual optima", ok, 0, 0);
}

static void
test_scaling_invariance()
{
    KnownLp k = make_known_lp(30, 40, 0.2, 77, false);
    Scaling sc = compute_scaling(k.p.A, 10, 1.0);
    LpProblem q = apply_scaling(k.p, sc);
    std::vector<double> xs(k.x), ys(k.y);
    for (size_t j = 0; j < xs.size(); ++j) xs[j] /= sc.col[j];
    for (size_t i = 0; i < ys.size(); ++i) ys[i] /= sc.row[i];
    LpSolution e = evaluate_solution(q, xs, ys);
    // The scaled pair must be optimal for the scaled problem with the same
    // objective, and scaling must actually balance the matrix.
    bool ok = e.primal_residual < 1e-9 && e.dual_residual < 1e-9 &&
              std::abs(e.primal_objective - k.obj) < 1e-9 && e.gap < 1e-7;
    tlog("SOLVER_SCALING", "scaled optimum stays optimal (same objective)", ok, 0, 0);
    double lo = 1e300, hi = 0;
    std::vector<double> rn(q.rows());
    Sparse::row_norms(q.A, rn.data(), Sparse::Norm::Linf);
    for (double v : rn) { lo = std::min(lo, v); hi = std::max(hi, v); }
    std::vector<double> rn0(k.p.rows());
    Sparse::row_norms(k.p.A, rn0.data(), Sparse::Norm::Linf);
    double lo0 = 1e300, hi0 = 0;
    for (double v : rn0) { lo0 = std::min(lo0, v); hi0 = std::max(hi0, v); }
    tlog("SOLVER_SCALING", "row norms are equilibrated", hi / lo <= hi0 / lo0 * 1.0001, 0, 0);
    LpSolution back;
    back.x = xs; back.y = ys;
    unscale_solution(back.x, back.y, sc);
    tlog("SOLVER_SCALING", "unscale_solution inverts the mapping",
        near_vec(back.x, k.x, 1e-12) && near_vec(back.y, k.y, 1e-12), 0, 0);
}

template <template <typename> class Store>
static void
test_lp_solves(const char *cat, LpMethod method = LpMethod::Pdlp)
{
    std::string t = std::string(std::is_same_v<Store<double>, Cpu::HostStorage<double>> ? "cpu" : "cuda") +
                    (method == LpMethod::Ipm ? " ipm" : method == LpMethod::Simplex ? " simplex" : "");
    struct Case { size_t m, n; double dens; unsigned seed; bool extras; };
    for (Case cs : {Case{8, 12, 0.4, 1, false}, Case{30, 45, 0.2, 2, false},
             Case{120, 160, 0.06, 3, false}, Case{200, 220, 0.05, 4, false},
             Case{30, 45, 0.2, 5, true}, Case{100, 140, 0.08, 6, true},
             Case{150, 200, 0.05, 7, true}}) {
        KnownLp k = make_known_lp(cs.m, cs.n, cs.dens, cs.seed, cs.extras);
        for (bool pre : {false, true}) {
            if (cs.extras && !pre) continue; // extras exist to exercise presolve
            SolverOptions o;
            o.set_tolerance(1e-6);
            o.presolve = pre;
            o.method = method;
            LpSolution s = solve_lp<Store>(k.p, o);
            std::string nm = t + " LP " + std::to_string(cs.m) + "x" +
                             std::to_string(cs.n) + (cs.extras ? " +extras" : "") +
                             (pre ? " presolve" : "");
            std::string det = std::string(to_string(s.status)) + " it=" +
                              std::to_string(s.iterations);
            tlog(cat, nm.c_str(), solution_ok(k.p, s, k.obj, 1e-6),
                std::abs(s.primal_objective - k.obj), s.seconds * 1000, det.c_str());
        }
    }
    // presolve really does something on the extras instances
    KnownLp kx = make_known_lp(100, 140, 0.08, 6, true);
    Presolve pre(kx.p);
    tlog(cat, (t + " presolve removes the injected rows/cols").c_str(),
        pre.removed_rows() >= kx.single_rows && pre.removed_cols() >= kx.fixed_cols,
        0, 0);

    // Wyndor Glass: max 3x + 5y, x <= 4, 2y <= 12, 3x + 2y <= 18.
    {
        const char *wyndor = R"(NAME WYNDOR
OBJSENSE
    MAX
ROWS
 N PROFIT
 L P1
 L P2
 L P3
COLUMNS
 X PROFIT 3 P1 1
 X P3 3
 Y PROFIT 5 P2 2
 Y P3 2
RHS
 RHS P1 4 P2 12
 RHS P3 18
ENDATA
)";
        std::istringstream in(wyndor);
        LpProblem p = read_mps(in);
        for (bool pre_on : {false, true}) {
            if (method == LpMethod::Ipm && !pre_on) continue;
            SolverOptions o;
            o.set_tolerance(1e-8);
            o.presolve = pre_on;
            o.method = method;
            LpSolution s = solve_lp<Store>(p, o);
            bool ok = s.status == Status::Optimal &&
                      near(s.x[0], 2, 1e-5) && near(s.x[1], 6, 1e-5) &&
                      near(s.primal_objective, -36, 1e-6) &&
                      std::abs(s.y[0]) < 1e-5 && near(s.y[1], -1.5, 1e-4) &&
                      near(s.y[2], -1.0, 1e-4);
            tlog(cat, (t + " Wyndor via MPS" + (pre_on ? " presolve" : "")).c_str(),
                ok, std::abs(s.primal_objective + 36), s.seconds * 1000,
                to_string(s.status));
        }
    }

    if (method == LpMethod::Ipm) return; // certificates are PDLP-only
    // Infeasible without presolve: x1 + x2 >= 4 with x1, x2 <= 1.
    {
        LpBuilder b(1, 2);
        b.a(0, 0, 1).a(0, 1, 1);
        b.p.c = {1, 1};
        b.p.row_lb = {4};
        b.p.col_ub = {1, 1};
        LpProblem p = b.build();
        SolverOptions o;
        o.presolve = false;
        LpSolution s = solve_lp<Store>(p, o);
        tlog(cat, (t + " detects infeasible").c_str(), s.status == Status::Infeasible,
            0, s.seconds * 1000, to_string(s.status));
    }
    // Unbounded: min -x1 - x2, x1 - x2 <= 1, x >= 0.
    {
        LpBuilder b(1, 2);
        b.a(0, 0, 1).a(0, 1, -1);
        b.p.c = {-1, -1};
        b.p.row_ub = {1};
        LpProblem p = b.build();
        SolverOptions o;
        o.presolve = false;
        LpSolution s = solve_lp<Store>(p, o);
        tlog(cat, (t + " detects unbounded").c_str(), s.status == Status::Unbounded,
            0, s.seconds * 1000, to_string(s.status));
    }
    // Presolve-detected statuses are reported through solve_lp as well.
    {
        LpBuilder b(2, 1);
        b.a(0, 0, 1);
        b.p.row_lb = {0, 1};
        LpProblem p = b.build();
        tlog(cat, (t + " presolve infeasible via solve_lp").c_str(),
            solve_lp<Store>(p).status == Status::Infeasible, 0, 0);
    }
    // Iteration limit is honoured.
    if (method == LpMethod::Pdlp) {
        KnownLp k = make_known_lp(120, 160, 0.06, 3, false);
        SolverOptions o;
        o.set_tolerance(1e-12);
        o.max_iterations = 200;
        o.presolve = false;
        LpSolution s = solve_lp<Store>(k.p, o);
        tlog(cat, (t + " iteration limit").c_str(),
            s.status == Status::IterationLimit && s.iterations <= 200 + o.check_frequency,
            0, 0);
    }
}

// ─── real models (tests/data, netlib) ────────────────────────────────────────
// Reference optima are HiGHS' values. Files are looked up relative to the
// repository root; the test is skipped if they are not found.
static void
test_netlib_models()
{
    struct Ref { const char *name; double obj; };
    const Ref refs[] = {{"afiro", -4.6475314286e+02}, {"avgas", -7.75},
        {"blending", -3200.0}, {"sctest", 5.75}, {"adlittle", 2.2549496316e+05}};
    const char *dirs[] = {"tests/data/", "../tests/data/", "data/"};
    for (const Ref &r : refs) {
        LpProblem p;
        bool found = false;
        for (const char *d : dirs) {
            try {
                p = read_mps_file(std::string(d) + r.name + ".mps");
                found = true;
                break;
            } catch (const std::exception &) {}
        }
        if (!found) {
            tlog("SOLVER_NETLIB", (std::string(r.name) + " (data file not found, skipped)").c_str(), true, 0, 0);
            continue;
        }
        const double sgn = p.maximize ? -1.0 : 1.0;
        for (LpMethod m : {LpMethod::Pdlp, LpMethod::Ipm, LpMethod::Simplex, LpMethod::Auto}) {
            SolverOptions o;
            o.method = m;
            o.set_tolerance(m == LpMethod::Pdlp ? 1e-6 : 1e-8);
            LpSolution s = solve_lp<Cpu::HostStorage>(p, o);
            const double tol = m == LpMethod::Pdlp ? 2e-5 : 1e-7;
            const double err = std::abs(sgn * s.primal_objective - r.obj) / (1 + std::abs(r.obj));
            const char *mn = m == LpMethod::Pdlp ? "pdlp" : m == LpMethod::Ipm ? "ipm"
                             : m == LpMethod::Simplex ? "simplex" : "auto";
            std::string det = std::string(to_string(s.status)) + " it=" + std::to_string(s.iterations);
            tlog("SOLVER_NETLIB", (std::string(r.name) + " " + mn).c_str(),
                s.status == Status::Optimal && err < tol, err, s.seconds * 1000, det.c_str());
        }
    }
    // (the crossover check is inside the loop above via the extra pass below)
    for (const Ref &r : refs) {
        LpProblem p;
        bool found = false;
        for (const char *d : dirs) {
            try { p = read_mps_file(std::string(d) + r.name + ".mps"); found = true; break; }
            catch (const std::exception &) {}
        }
        if (!found) continue;
        const double sgn = p.maximize ? -1.0 : 1.0;
        SolverOptions o;
        o.method = LpMethod::Ipm;
        o.crossover = true;
        o.set_tolerance(1e-8);
        LpSolution s = solve_lp<Cpu::HostStorage>(p, o);
        const double err = std::abs(sgn * s.primal_objective - r.obj) / (1 + std::abs(r.obj));
        tlog("SOLVER_NETLIB", (std::string(r.name) + " ipm+crossover").c_str(),
            s.status == Status::Optimal && err < 1e-7, err, s.seconds * 1000, to_string(s.status));
        { // PDLP with polishing to a tight tolerance
            SolverOptions op;
            op.method = LpMethod::Pdlp;
            op.pdlp_polish = true;
            op.set_tolerance(1e-8);
            LpSolution sp = solve_lp<Cpu::HostStorage>(p, op);
            const double ep = std::abs(sgn * sp.primal_objective - r.obj) / (1 + std::abs(r.obj));
            tlog("SOLVER_NETLIB", (std::string(r.name) + " pdlp polish 1e-8").c_str(),
                sp.status == Status::Optimal && ep < 1e-6, ep, sp.seconds * 1000, to_string(sp.status));
        }
        for (int nm : {1, -1}) { // forced normal equations / forced augmented system
            SolverOptions on;
            on.method = LpMethod::Ipm;
            on.ipm_normal = nm;
            on.set_tolerance(1e-8);
            LpSolution sn = solve_lp<Cpu::HostStorage>(p, on);
            const double en = std::abs(sgn * sn.primal_objective - r.obj) / (1 + std::abs(r.obj));
            tlog("SOLVER_NETLIB", (std::string(r.name) + (nm == 1 ? " ipm normal equations" : " ipm augmented")).c_str(),
                sn.status == Status::Optimal && en < 1e-7, en, sn.seconds * 1000, to_string(sn.status));
        }
    }
    // An infeasible netlib model: PDLP (and Auto, which falls back to it)
    // report infeasibility; the interior-point method does not converge.
    try {
        LpProblem p = read_mps_file("tests/data/galenet.mps");
        for (LpMethod m : {LpMethod::Pdlp, LpMethod::Simplex, LpMethod::Auto}) {
            SolverOptions o;
            o.method = m;
            LpSolution s = solve_lp<Cpu::HostStorage>(p, o);
            tlog("SOLVER_NETLIB", (std::string("galenet infeasible ") + (m == LpMethod::Pdlp ? "pdlp" : m == LpMethod::Simplex ? "simplex" : "auto")).c_str(),
                s.status == Status::Infeasible, 0, s.seconds * 1000, to_string(s.status));
        }
        SolverOptions o;
        o.method = LpMethod::Ipm;
        LpSolution s = solve_lp<Cpu::HostStorage>(p, o);
        tlog("SOLVER_NETLIB", "galenet ipm does not report optimal",
            s.status != Status::Optimal, 0, s.seconds * 1000, to_string(s.status));
    } catch (const std::exception &) {
        tlog("SOLVER_NETLIB", "galenet (data file not found, skipped)", true, 0, 0);
    }
}

// ─── dual simplex specifics ─────────────────────────────────────────────────
// Basis factorization with eta updates and periodic refactorization, against a
// dense solve.
static void
test_basis_factor()
{
    std::mt19937 g(11);
    std::uniform_real_distribution<double> u(-1, 1);
    bool ok = true;
    double worst = 0;
    for (int m : {3, 8, 30, 100}) {
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                if (u(g) > 0.6 || i == j) B(i, j) = u(g);
        for (int i = 0; i < m; ++i) B(i, i) += 3;
        std::vector<std::vector<int>> I(m);
        std::vector<std::vector<double>> V(m);
        std::vector<BasisFactor::Column> C(m);
        auto factor_all = [&](BasisFactor &bf) {
            for (int j = 0; j < m; ++j) {
                I[j].clear(); V[j].clear();
                for (int i = 0; i < m; ++i)
                    if (B(i, j) != 0) { I[j].push_back(i); V[j].push_back(B(i, j)); }
                C[j] = {I[j].data(), V[j].data(), static_cast<int>(I[j].size())};
            }
            std::vector<int> sing, freerow;
            return bf.factor(m, C, sing, freerow);
        };
        BasisFactor bf;
        ok = ok && factor_all(bf) == 0;
        for (int upd = 0; upd < 40; ++upd) {
            Eigen::VectorXd b(m);
            for (int i = 0; i < m; ++i) b(i) = u(g);
            std::vector<double> x(b.data(), b.data() + m), y(b.data(), b.data() + m);
            bf.ftran(x);
            bf.btran(y);
            Eigen::VectorXd xr = B.partialPivLu().solve(b), yr = B.transpose().partialPivLu().solve(b);
            double e = 0;
            for (int i = 0; i < m; ++i)
                e = std::max({e, std::abs(x[i] - xr(i)), std::abs(y[i] - yr(i))});
            worst = std::max(worst, e);
            ok = ok && e < 1e-6;
            { // hypersparse solves agree with the dense ones
                std::vector<double> bs(m, 0.0);
                SVec sf, sb;
                sf.init(m); sb.init(m);
                const int nnz = 1 + static_cast<int>(g() % 3);
                for (int t = 0; t < nnz; ++t) {
                    const int i = static_cast<int>(g() % m);
                    bs[i] = u(g);
                }
                for (int i = 0; i < m; ++i)
                    if (bs[i] != 0.0) { sf.set(i, bs[i]); sb.set(i, bs[i]); }
                std::vector<double> xd = bs, yd = bs;
                bf.ftran(xd);
                bf.btran(yd);
                bf.ftran(sf);
                bf.btran(sb);
                double es = 0;
                for (int i = 0; i < m; ++i)
                    es = std::max({es, std::abs(sf.v[i] - xd[i]), std::abs(sb.v[i] - yd[i])});
                // pattern must cover every nonzero
                std::vector<char> seen(m, 0);
                for (int i : sf.idx) seen[i] = 1;
                for (int i = 0; i < m; ++i)
                    if (xd[i] != 0.0 && !seen[i]) es = 1;
                std::fill(seen.begin(), seen.end(), 0);
                for (int i : sb.idx) seen[i] = 1;
                for (int i = 0; i < m; ++i)
                    if (yd[i] != 0.0 && !seen[i]) es = 1;
                worst = std::max(worst, es);
                ok = ok && es < 1e-9;
            }
            const int r = static_cast<int>(g() % m);
            Eigen::VectorXd a(m);
            for (int i = 0; i < m; ++i) a(i) = (u(g) > 0.5) ? u(g) : 0.0;
            a(g() % m) += 2;
            std::vector<double> alpha(a.data(), a.data() + m);
            bf.ftran(alpha);
            if (std::abs(alpha[r]) < 1e-2) continue;
            B.col(r) = a;
            if (!bf.update(r, alpha) || bf.updates() >= 12) ok = ok && factor_all(bf) == 0; // refactor cycle
        }
    }
    tlog("SOLVER_SIMPLEX", "basis LU: ftran/btran with eta updates and refactorization", ok, worst, 0);
    // larger sparse bases (slack-heavy, like simplex bases), some with dependent
    // columns: after replacing each reported column by the slack of its free
    // row the factorization is exact
    {
        bool ok2 = true;
        double worst2 = 0;
        for (int trial = 0; trial < 12; ++trial) {
            const int m = 50 + 40 * trial;
            std::uniform_int_distribution<int> row(0, m - 1);
            std::vector<std::vector<int>> I(m);
            std::vector<std::vector<double>> V(m);
            for (int j = 0; j < m; ++j) {
                if (u(g) < 0.3) { I[j] = {j}; V[j] = {1.0}; continue; } // slack
                const int nz = 1 + static_cast<int>(g() % 6);
                std::vector<int> rows;
                for (int t = 0; t < nz; ++t) rows.push_back(row(g));
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                for (int r : rows) { I[j].push_back(r); V[j].push_back(u(g) * 10); }
            }
            if (trial % 3 == 2) { I[m - 1] = I[0]; V[m - 1] = V[0]; } // dependent column
            BasisFactor bf;
            std::vector<int> sing, freerow;
            for (int round = 0; round < 10; ++round) {
                std::vector<BasisFactor::Column> C(m);
                for (int j = 0; j < m; ++j) C[j] = {I[j].data(), V[j].data(), static_cast<int>(I[j].size())};
                if (bf.factor(m, C, sing, freerow) == 0) break;
                for (size_t t = 0; t < sing.size(); ++t) {
                    I[sing[t]] = {freerow[t]};
                    V[sing[t]] = {1.0};
                }
            }
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m, m);
            for (int j = 0; j < m; ++j)
                for (size_t t = 0; t < I[j].size(); ++t) B(I[j][t], j) += V[j][t];
            Eigen::VectorXd b(m);
            for (int i = 0; i < m; ++i) b(i) = u(g);
            std::vector<double> x(b.data(), b.data() + m), y(b.data(), b.data() + m);
            bf.ftran(x);
            bf.btran(y);
            Eigen::VectorXd xv = Eigen::Map<Eigen::VectorXd>(x.data(), m);
            Eigen::VectorXd yv = Eigen::Map<Eigen::VectorXd>(y.data(), m);
            const double e = std::max((B * xv - b).norm(), (B.transpose() * yv - b).norm()) /
                             (1 + b.norm());
            worst2 = std::max(worst2, e);
            ok2 = ok2 && e < 1e-9;
        }
        tlog("SOLVER_SIMPLEX", "basis LU: sparse bases with dependent columns (residual)", ok2, worst2, 0);
    }
    // a singular basis is reported, with a replacement row
    {
        std::vector<int> i0{0, 1}, i1{0, 1}, i2{2};
        std::vector<double> v0{1, 2}, v1{2, 4}, v2{1};
        std::vector<BasisFactor::Column> cs{{i0.data(), v0.data(), 2}, {i1.data(), v1.data(), 2}, {i2.data(), v2.data(), 1}};
        BasisFactor bf2;
        std::vector<int> sing, freerow;
        int ns = bf2.factor(3, cs, sing, freerow);
        tlog("SOLVER_SIMPLEX", "basis LU reports a singular column", ns == 1 && sing.size() == 1 && freerow.size() == 1, 0, 0);
    }
}

// Warm start: after a bound change the old basis re-solves in fewer
// iterations and gives the cold-start answer.
static void
test_simplex_warm_start()
{
    bool ok = true;
    long warm_it = 0, cold_it = 0;
    for (unsigned seed = 1; seed <= 4; ++seed) {
        KnownLp k = make_known_lp(60, 80, 0.1, 300 + seed, false);
        SolverOptions o;
        o.set_tolerance(1e-9);
        SimplexBasis basis;
        DualSimplex first;
        LpSolution s0 = first.solve(k.p, o, nullptr, &basis);
        if (s0.status != Status::Optimal) { ok = false; continue; }
        // tighten the bound of a column that is strictly inside its range
        LpProblem q = k.p;
        size_t jbest = 0;
        double gap = -1;
        for (size_t j = 0; j < q.cols(); ++j) {
            double room = std::min(s0.x[j] - q.col_lb[j], q.col_ub[j] - s0.x[j]);
            if (std::isfinite(room) && room > gap && std::isfinite(q.col_ub[j])) { gap = room; jbest = j; }
        }
        if (gap <= 0) continue;
        q.col_ub[jbest] = s0.x[jbest] - 0.5 * gap;
        if (q.col_lb[jbest] > q.col_ub[jbest]) continue;
        DualSimplex w, c;
        LpSolution sw = w.solve(q, o, &basis, nullptr);
        LpSolution sc = c.solve(q, o);
        ok = ok && sw.status == Status::Optimal && sc.status == Status::Optimal &&
             std::abs(sw.primal_objective - sc.primal_objective) <= 1e-7 * (1 + std::abs(sc.primal_objective));
        warm_it += sw.iterations;
        cold_it += sc.iterations;
    }
    tlog("SOLVER_SIMPLEX", "warm start after a bound change matches the cold solve", ok, 0, 0,
        (std::to_string(warm_it) + " warm vs " + std::to_string(cold_it) + " cold iterations").c_str());
    tlog("SOLVER_SIMPLEX", "warm start needs fewer iterations", warm_it < cold_it, 0, 0);
}

// Degenerate transportation problems: simplex vs interior point.
static void
test_simplex_degenerate()
{
    std::mt19937 g(21);
    bool ok = true;
    double worst = 0;
    for (int trial = 0; trial < 6; ++trial) {
        const size_t S = 8 + trial, D = 9 + trial;
        std::uniform_int_distribution<int> cost(1, 20), amt(3, 9);
        std::vector<double> sup(S), dem(D);
        long tot = 0;
        for (auto &v : sup) { v = amt(g); tot += (long)v; }
        long dt = 0;
        for (auto &v : dem) { v = amt(g); dt += (long)v; }
        for (size_t j = 0; dt != tot; j = (j + 1) % D) { // balance the totals
            double step = dt < tot ? 1 : -1;
            if (dem[j] + step >= 1) { dem[j] += step; dt += (long)step; }
        }
        LpBuilder b(S + D, S * D);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < D; ++j) {
                b.a(i, i * D + j, 1).a(S + j, i * D + j, 1);
                b.p.c[i * D + j] = cost(g);
            }
        for (size_t i = 0; i < S; ++i) b.p.row_lb[i] = b.p.row_ub[i] = sup[i];
        for (size_t j = 0; j < D; ++j) b.p.row_lb[S + j] = b.p.row_ub[S + j] = dem[j];
        LpProblem p = b.build();
        SolverOptions os, oi;
        os.method = LpMethod::Simplex;
        os.set_tolerance(1e-9);
        oi.method = LpMethod::Ipm;
        oi.set_tolerance(1e-9);
        LpSolution a = solve_lp<Cpu::HostStorage>(p, os), c = solve_lp<Cpu::HostStorage>(p, oi);
        const double e = std::abs(a.primal_objective - c.primal_objective) / (1 + std::abs(c.primal_objective));
        worst = std::max(worst, e);
        ok = ok && a.status == Status::Optimal && c.status == Status::Optimal && e < 1e-7 &&
             a.primal_residual < 1e-8;
    }
    tlog("SOLVER_SIMPLEX", "degenerate transportation problems: simplex == interior point", ok, worst, 0);
}

// Concurrent Auto agrees with the sequential strategy; a preset stop flag
// interrupts every method.
template <template <typename> class Store>
static void
test_concurrent_auto(const char *cat)
{
    std::mt19937 g(33);
    bool ok = true;
    double worst = 0;
    for (int trial = 0; trial < 4; ++trial) {
        const size_t S = 10 + 3 * trial, D = 12 + 2 * trial;
        std::uniform_int_distribution<int> cost(1, 30), amt(2, 12);
        LpBuilder b(S + D, S * D);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = 0; j < D; ++j) {
                b.a(i, i * D + j, 1).a(S + j, i * D + j, 1);
                b.p.c[i * D + j] = cost(g);
            }
        for (size_t i = 0; i < S; ++i) { b.p.row_lb[i] = -kInf; b.p.row_ub[i] = amt(g) + 6; }
        for (size_t j = 0; j < D; ++j) { b.p.row_lb[S + j] = amt(g); b.p.row_ub[S + j] = kInf; }
        LpProblem p = b.build();
        SolverOptions oc, os;
        oc.method = os.method = LpMethod::Auto;
        oc.set_tolerance(1e-8);
        os.set_tolerance(1e-8);
        os.concurrent = false;
        LpSolution a = solve_lp<Store>(p, oc), c = solve_lp<Store>(p, os);
        const double e = std::abs(a.primal_objective - c.primal_objective) / (1 + std::abs(c.primal_objective));
        worst = std::max(worst, e);
        ok = ok && a.status == c.status && (a.status != Status::Optimal || e < 1e-7) &&
             a.duals_verified && c.duals_verified;
    }
    tlog(cat, "concurrent Auto == sequential Auto", ok, worst, 0);
    // infeasible: the simplex certificate wins the race
    {
        LpBuilder b(2, 2);
        b.a(0, 0, 1).a(0, 1, 1).a(1, 0, 1).a(1, 1, 1);
        b.p.c = {1, 1};
        b.p.row_lb = {3, -kInf};
        b.p.row_ub = {kInf, 1};
        LpProblem p = b.build();
        SolverOptions o;
        o.method = LpMethod::Auto;
        o.presolve = false;
        LpSolution s = solve_lp<Store>(p, o);
        tlog(cat, "concurrent Auto detects infeasibility", s.status == Status::Infeasible, 0, 0,
            to_string(s.status));
    }
    // a stop flag that is already set: every method returns Interrupted
    {
        LpBuilder b(3, 4);
        b.a(0, 0, 1).a(0, 1, 2).a(1, 1, 1).a(1, 2, 3).a(2, 2, 1).a(2, 3, 1).a(2, 0, 1);
        b.p.c = {1, 2, 3, 1};
        b.p.row_lb = {2, 1, 1};
        b.p.row_ub = {kInf, kInf, kInf};
        LpProblem p = b.build();
        StopFlag stop;
        stop.set();
        bool all = true;
        for (LpMethod m : {LpMethod::Pdlp, LpMethod::Ipm, LpMethod::Simplex, LpMethod::Auto}) {
            SolverOptions o;
            o.method = m;
            o.presolve = false;
            o.interrupt = &stop;
            o.check_frequency = 1;
            LpSolution s = solve_lp<Store>(p, o);
            all = all && s.status == Status::Interrupted;
        }
        tlog(cat, "stop flag interrupts every method", all, 0, 0);
    }
}

// ─── MILP ───────────────────────────────────────────────────────────────────
// Reference values come from enumeration (pure integer problems) or enumeration
// of the integer part plus an LP for the continuous part (mixed problems).
static void
test_milp()
{
    const char *cat = "SOLVER_MILP";
    auto ints = [](LpProblem &p, std::initializer_list<int> cols) {
        p.is_integer.assign(p.cols(), 0);
        for (int j : cols) p.is_integer[j] = 1;
    };
    // the classic textbook problem: max 5x + 4y + 3z, optimum 13 at (2, 0, 1)
    {
        LpBuilder b(3, 3);
        b.a(0, 0, 2).a(0, 1, 3).a(0, 2, 1).a(1, 0, 4).a(1, 1, 1).a(1, 2, 2)
            .a(2, 0, 3).a(2, 1, 4).a(2, 2, 2);
        b.p.c = {-5, -4, -3};
        b.p.row_ub = {5, 11, 8};
        b.p.col_ub = {kInf, kInf, kInf};
        LpProblem p = b.build();
        ints(p, {0, 1, 2});
        MilpSolution s = solve_milp(p);
        tlog(cat, "textbook integer program", s.status == MilpStatus::Optimal && near(s.objective, -13) &&
            near_vec(s.x, {2, 0, 1}), std::abs(s.objective + 13), s.seconds * 1000,
            to_string(s.status));
    }
    // infeasible only through integrality: 2x = 1
    {
        LpBuilder b(1, 1);
        b.a(0, 0, 2);
        b.p.c = {1};
        b.p.row_lb = b.p.row_ub = {1};
        b.p.col_ub = {10};
        LpProblem p = b.build();
        ints(p, {0});
        MilpSolution s = solve_milp(p);
        tlog(cat, "integer infeasible (2x = 1)", s.status == MilpStatus::Infeasible, 0, 0, to_string(s.status));
    }
    // LP relaxation unbounded
    {
        LpBuilder b(1, 2);
        b.a(0, 0, 1).a(0, 1, -1);
        b.p.c = {-1, -1};
        b.p.row_ub = {3};
        b.p.col_ub = {kInf, kInf};
        LpProblem p = b.build();
        ints(p, {0, 1});
        MilpSolution s = solve_milp(p);
        tlog(cat, "unbounded relaxation reported", s.status == MilpStatus::Unbounded, 0, 0, to_string(s.status));
    }
    // random multi-dimensional 0/1 knapsacks against enumeration
    {
        std::mt19937 g(91);
        std::uniform_int_distribution<int> w(1, 20), v(1, 30);
        bool ok = true;
        double worst = 0;
        long nodes = 0;
        for (int trial = 0; trial < 25; ++trial) {
            const int n = 14, m = 3;
            LpBuilder b(m, n);
            std::vector<std::vector<int>> A(m, std::vector<int>(n));
            std::vector<int> cap(m), val(n);
            for (int j = 0; j < n; ++j) val[j] = v(g);
            for (int i = 0; i < m; ++i) {
                int tot = 0;
                for (int j = 0; j < n; ++j) { A[i][j] = w(g); tot += A[i][j]; b.a(i, j, A[i][j]); }
                cap[i] = tot / 3;
                b.p.row_ub[i] = cap[i];
            }
            for (int j = 0; j < n; ++j) { b.p.c[j] = -val[j]; b.p.col_ub[j] = 1; }
            LpProblem p = b.build();
            ints(p, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
            long best = 0;
            for (int mask = 0; mask < (1 << n); ++mask) {
                bool f = true;
                for (int i = 0; i < m && f; ++i) {
                    int s = 0;
                    for (int j = 0; j < n; ++j) if (mask >> j & 1) s += A[i][j];
                    f = s <= cap[i];
                }
                if (!f) continue;
                long s = 0;
                for (int j = 0; j < n; ++j) if (mask >> j & 1) s += val[j];
                best = std::max(best, s);
            }
            MilpOptions mo;
            mo.mip_gap = 0; mo.abs_gap = 1e-9;
            MilpSolution s = solve_milp(p, mo);
            nodes += s.nodes;
            const double e = std::abs(s.objective + static_cast<double>(best));
            worst = std::max(worst, e);
            ok = ok && s.status == MilpStatus::Optimal && e < 1e-6;
        }
        tlog(cat, "25 random knapsacks == enumeration", ok, worst, 0, (std::to_string(nodes) + " nodes").c_str());
    }
    // random general-integer problems (incl. equalities, some infeasible) against enumeration
    {
        std::mt19937 g(92);
        std::uniform_int_distribution<int> coef(-4, 6), cost(-6, 6), rhs(2, 14);
        bool ok = true;
        int feasible_count = 0, infeasible_count = 0;
        for (int trial = 0; trial < 60; ++trial) {
            const int n = 5, m = 3, R = 3; // x_j in [0, R]
            LpBuilder b(m, n);
            std::vector<std::vector<int>> A(m, std::vector<int>(n));
            std::vector<int> c(n), lo(m), hi(m);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) { A[i][j] = coef(g); if (A[i][j]) b.a(i, j, A[i][j]); }
                const int r = rhs(g);
                if (i == 0 && trial % 3 == 0) { lo[i] = hi[i] = r; } // an equality
                else { lo[i] = -1000; hi[i] = r; }
                b.p.row_lb[i] = lo[i] == -1000 ? -kInf : lo[i];
                b.p.row_ub[i] = hi[i];
            }
            for (int j = 0; j < n; ++j) { c[j] = cost(g); b.p.c[j] = c[j]; b.p.col_ub[j] = R; }
            LpProblem p = b.build();
            ints(p, {0, 1, 2, 3, 4});
            long best = 1L << 40;
            std::vector<int> x(n);
            for (int code = 0; code < 1024; ++code) {
                int t = code;
                for (int j = 0; j < n; ++j) { x[j] = t % 4; t /= 4; }
                bool f = true;
                for (int i = 0; i < m && f; ++i) {
                    int s = 0;
                    for (int j = 0; j < n; ++j) s += A[i][j] * x[j];
                    f = s <= hi[i] && s >= lo[i];
                }
                if (!f) continue;
                long s = 0;
                for (int j = 0; j < n; ++j) s += c[j] * x[j];
                best = std::min(best, s);
            }
            MilpOptions mo;
            mo.mip_gap = 0; mo.abs_gap = 1e-9;
            MilpSolution s = solve_milp(p, mo);
            if (best == (1L << 40)) {
                ++infeasible_count;
                ok = ok && s.status == MilpStatus::Infeasible;
            } else {
                ++feasible_count;
                ok = ok && s.status == MilpStatus::Optimal && std::abs(s.objective - best) < 1e-6;
            }
        }
        tlog(cat, "60 random integer programs == enumeration", ok, 0, 0,
            (std::to_string(feasible_count) + " feasible, " + std::to_string(infeasible_count) + " infeasible").c_str());
    }
    // mixed problems: enumerate the integer part, solve the LP for the continuous part
    {
        std::mt19937 g(93);
        std::uniform_int_distribution<int> cost(-5, 5), rhs(3, 12);
        std::uniform_real_distribution<double> coef(-2, 3);
        bool ok = true;
        double worst = 0;
        int solved = 0;
        for (int trial = 0; trial < 30; ++trial) {
            const int ni = 3, nc = 3, n = ni + nc, m = 4;
            LpBuilder b(m, n);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) if (g() % 3) b.a(i, j, std::round(coef(g) * 4) / 4);
                b.p.row_ub[i] = rhs(g);
            }
            for (int j = 0; j < n; ++j) { b.p.c[j] = cost(g); b.p.col_ub[j] = j < ni ? 3 : 5; }
            LpProblem p = b.build();
            ints(p, {0, 1, 2});
            double best = kInf;
            for (int code = 0; code < 64; ++code) {
                LpProblem q = p;
                q.is_integer.clear();
                int t = code;
                for (int j = 0; j < ni; ++j) { q.col_lb[j] = q.col_ub[j] = t % 4; t /= 4; }
                SolverOptions so;
                so.method = LpMethod::Simplex;
                so.presolve = false;
                so.set_tolerance(1e-10);
                LpSolution s = solve_lp<Cpu::HostStorage>(q, so);
                if (s.status == Status::Optimal) best = std::min(best, s.primal_objective);
            }
            MilpOptions mo;
            mo.mip_gap = 0; mo.abs_gap = 1e-9;
            MilpSolution s = solve_milp(p, mo);
            if (!std::isfinite(best)) { ok = ok && s.status == MilpStatus::Infeasible; continue; }
            ++solved;
            const double e = std::abs(s.objective - best) / (1 + std::abs(best));
            worst = std::max(worst, e);
            ok = ok && s.status == MilpStatus::Optimal && e < 1e-6;
        }
        tlog(cat, "30 random mixed programs == enumeration + LP", ok, worst, 0,
            (std::to_string(solved) + " feasible").c_str());
    }
    // presolved-free search features switched off give the same answer
    {
        LpBuilder b(3, 3);
        b.a(0, 0, 2).a(0, 1, 3).a(0, 2, 1).a(1, 0, 4).a(1, 1, 1).a(1, 2, 2)
            .a(2, 0, 3).a(2, 1, 4).a(2, 2, 2);
        b.p.c = {-5, -4, -3};
        b.p.row_ub = {5, 11, 8};
        b.p.col_ub = {kInf, kInf, kInf};
        LpProblem p = b.build();
        ints(p, {0, 1, 2});
        MilpOptions mo;
        mo.propagate = false;
        mo.diving = false;
        MilpSolution s = solve_milp(p, mo);
        tlog(cat, "same answer without propagation and diving", s.status == MilpStatus::Optimal && near(s.objective, -13), 0, 0);
    }
}

int
main(int argc, char *argv[])
{
    headless = parse_headless(argc, argv);
    if (headless) printf("--- tensorPanini solver test suite ---\n");
    for (int run = 0; run < RUNS; ++run) {
        if (headless) printf("\n═══ RUN %d/%d ═══\n", run + 1, RUNS);
        test_mps_read();
        test_mps_edge_cases();
        test_mps_roundtrip();
        test_evaluate();
        test_presolve_singleton();
        test_presolve_other();
        test_presolve_duplicates();
        test_presolve_parallel();
        test_presolve_random();
        test_known_generator();
        test_scaling_invariance();
        test_netlib_models();
        test_basis_factor();
        test_simplex_warm_start();
        test_simplex_degenerate();
        test_concurrent_auto<Cpu::HostStorage>("SOLVER_AUTO_CPU");
        test_milp();
#ifdef PANINI_ENABLE_CUDA
        test_concurrent_auto<Cuda::CudaStorage>("SOLVER_AUTO_CUDA");
#endif
        test_lp_solves<Cpu::HostStorage>("SOLVER_SIMPLEX_CPU", LpMethod::Simplex);
        test_lp_solves<Cpu::HostStorage>("SOLVER_LP_CPU");
        test_lp_solves<Cpu::HostStorage>("SOLVER_IPM_CPU", LpMethod::Ipm);
#ifdef PANINI_ENABLE_CUDA
        test_lp_solves<Cuda::CudaStorage>("SOLVER_LP_CUDA");
        test_lp_solves<Cuda::CudaStorage>("SOLVER_IPM_CUDA", LpMethod::Ipm);
#endif
    }
    print_summary();
    int fail = 0;
    for (auto &r : results)
        if (!r.pass) fail++;
    return fail > 0 ? 1 : 0;
}
