// SPDX-License-Identifier: BSD-3-Clause
//
// LP presolve with a postsolve stack that restores primal AND dual solutions.
//
// The problem is held as a dynamic sparse matrix (entries linked from both
// their row and their column) because substitutions change coefficients and
// create fill. Reductions, applied until nothing changes:
//   - empty rows and free rows          (y = 0)
//   - empty columns                     (fixed at the bound the cost prefers;
//                                        may prove dual infeasibility)
//   - fixed columns                     (lb == ub: moved into the row bounds)
//   - singleton rows                    (turned into column bounds)
//   - redundant rows                    (activity bounds inside the row bounds)
//   - forcing rows                      (activity bound equals a row bound: every
//                                        column is fixed at the bound attaining it)
//   - dual fixing / dominated columns   (no row blocks moving x_j in the direction
//                                        its cost prefers: fixed at that bound)
//   - implied-free column singletons in equations (column and row substituted out)
//   - free, zero-cost column singletons in inequalities (row and column dropped)
//   - doubleton equations               (a x_j + b x_k = r: x_k substituted out)
//   - aggregation                       (implied-free columns with a few entries
//                                        substituted out of an equation, limited fill)
//   - duplicate / parallel rows         (merged; their bounds intersected)
//
// Every reduction stores what its postsolve needs AS IT WAS WHEN APPLIED (the
// costs and coefficients of later problems differ from the original ones).
// postsolve() runs the stack backwards: each step extends an optimal solution
// of the smaller problem to one of the problem before the reduction (x from
// the reduction's equation, y chosen so the reduced costs of the restored
// columns have the right signs). z = c - A^T y is recomputed on the original
// matrix at the end.
#pragma once

#include "solver/model.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Panini {
namespace Solver {

struct PresolveOptions {
    bool empty_rows = true;
    bool empty_cols = true;
    bool fixed_cols = true;
    bool singleton_rows = true;
    bool duplicate_rows = true;
    bool activity_rows = true;       // redundant and forcing rows
    bool dual_fixing = true;         // dominated columns
    bool singleton_cols = true;      // implied-free / free column singletons
    bool doubleton_equations = true;
    bool aggregate = true;           // implied-free columns substituted out of equations
    bool parallel_cols = true;       // proportional columns with proportional costs merged
    // MIP mode: columns marked in LpProblem::is_integer are kept integral (bounds
    // rounded inward; only continuous columns are substituted or merged away). The
    // reduced problem keeps the marks. Dual recovery in postsolve is NOT valid then
    // (only x is), and coefficient tightening (which changes the LP but not the
    // integer-feasible set) may be enabled.
    bool respect_integrality = false;
    bool coef_tightening = false;
    double feas_tol = 1e-9;
};

class Presolve {
  public:
    Presolve(const LpProblem &p, const PresolveOptions &opt = {})
        : orig_(p), opt_(opt)
    {
        run();
    }

    // Outcome of presolve itself: NotSolved (normal), Infeasible or
    // Unbounded (proven during presolve; no solve needed).
    Status status() const { return status_; }
    const LpProblem &reduced() const { return reduced_; }
    size_t removed_rows() const { return orig_.rows() - reduced_.rows(); }
    long coefficients_tightened() const { return coef_tightened_; }
    size_t removed_cols() const { return orig_.cols() - reduced_.cols(); }

    // Build the original-space solution from a reduced-space one. Also works
    // when the reduced problem is empty (pass an empty LpSolution with
    // status Optimal).
    LpSolution
    postsolve(const LpSolution &r) const
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        std::vector<double> x(n, 0.0), y(m, 0.0);
        for (size_t j = 0; j < col_map_.size(); ++j)
            if (j < r.x.size()) x[col_map_[j]] = r.x[j];
        for (size_t i = 0; i < row_map_.size(); ++i)
            if (i < r.y.size()) y[row_map_[i]] = r.y[i];

        // reduced cost of column j in the problem the op was applied to:
        // cost then minus the column then (rows other than the op's row)
        auto zrest = [&](double c, const std::vector<std::pair<int, double>> &col) {
            double z = c;
            for (const auto &e : col) z -= e.second * y[e.first];
            return z;
        };
        for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
            const Op &op = *it;
            switch (op.kind) {
            case Op::EmptyRow: y[op.i] = 0; break;
            case Op::EmptyCol:
            case Op::FixedCol: x[op.j] = op.value; break;
            case Op::SingletonRow: {
                const double zj = zrest(op.cj, op.colj);
                if ((zj > 0 && op.tight_lb) || (zj < 0 && op.tight_ub)) y[op.i] = zj / op.a;
                break;
            }
            case Op::DuplicateRow: {
                const double yi = y[op.i];
                if (yi > 0) {
                    if (!(op.lb_i >= op.impl_lb_k)) { y[op.k] = yi / op.alpha; y[op.i] = 0; }
                } else if (yi < 0) {
                    if (!(op.ub_i <= op.impl_ub_k)) { y[op.k] = yi / op.alpha; y[op.i] = 0; }
                }
                break;
            }
            case Op::Forcing: {
                // every column of the row sits at the bound that attains the
                // row's extreme activity; y_i gives each the right reduced cost
                double yi = 0;
                for (const auto &fc : op.forced) {
                    const double v = zrest(fc.c, fc.col) / fc.a;
                    yi = op.at_upper ? std::min(yi, v) : std::max(yi, v);
                }
                y[op.i] = yi;
                break;
            }
            case Op::ImpliedFreeSingleton: {
                double s = op.rhs;
                for (const auto &e : op.row) s -= e.second * x[e.first];
                x[op.j] = s / op.a;
                y[op.i] = zrest(op.cj, op.colj) / op.a; // z_j = 0: x_j is inside its bounds
                break;
            }
            case Op::FreeSingletonIneq: {
                double act = 0;
                for (const auto &e : op.row) act += e.second * x[e.first];
                // a x_j in [lb - act, ub - act]: the value closest to 0
                double lo = op.lb_i - act, hi = op.ub_i - act;
                double v = 0;
                if (lo > 0) v = lo;
                else if (hi < 0) v = hi;
                x[op.j] = v / op.a;
                y[op.i] = 0;
                break;
            }
            case Op::Doubleton: {
                // a x_j + b x_k = rhs
                x[op.k] = (op.rhs - op.a * x[op.j]) / op.b;
                // The reduced problem's reduced cost of x_j, zt = zj - (a/b) zk, says
                // which of x_j's reduced bounds binds (zt > 0: lower, zt < 0: upper).
                // If that bound is x_j's own, keep zt on x_j (z_k = 0); if it was
                // implied through x_k, x_k is the variable at a bound: z_j = 0 and
                // z_k = -(b/a) zt. Deciding by signs only (not by the position of
                // x_j) also works for interior-point solutions.
                const double zj = zrest(op.cj, op.colj), zk = zrest(op.ck, op.colk);
                const double zt = zj - op.a / op.b * zk;
                const bool implied = (zt > 0 && op.tight_lb) || (zt < 0 && op.tight_ub);
                y[op.i] = implied ? zj / op.a : zk / op.b;
                break;
            }
            case Op::ParallelCol: {
                // x[j] holds w = x_j + alpha x_k. Split it inside both columns'
                // bounds: x_j in [lb_i, ub_i], t = alpha x_k in [impl_lb_k, impl_ub_k].
                // When w is at a bound the interval is one point, which is exactly
                // what the (proportional) reduced costs of both columns require.
                const double w = x[op.j];
                const double lo = std::max(op.lb_i, w - op.impl_ub_k);
                const double hi = std::min(op.ub_i, w - op.impl_lb_k);
                const double xj = std::isfinite(lo) ? lo : std::isfinite(hi) ? hi : 0.0;
                x[op.j] = xj;
                x[op.k] = (w - xj) / op.alpha;
                break;
            }
            }
        }
        LpSolution e = evaluate_solution(orig_, x, y);
        e.status = r.status;
        e.iterations = r.iterations;
        e.seconds = r.seconds;
        return e;
    }

  private:
    using Sparse1 = std::vector<std::pair<int, double>>;
    struct ForcedCol {
        int j;
        double a, c;
        Sparse1 col; // other rows
    };
    struct Op {
        enum Kind {
            EmptyRow, EmptyCol, FixedCol, SingletonRow, DuplicateRow, Forcing,
            ImpliedFreeSingleton, FreeSingletonIneq, Doubleton, ParallelCol
        } kind;
        int i = -1, j = -1, k = -1;
        double value = 0, a = 0, b = 0, alpha = 0, rhs = 0, cj = 0, ck = 0;
        bool tight_lb = false, tight_ub = false, at_upper = false;
        double lb_i = 0, ub_i = 0, impl_lb_k = 0, impl_ub_k = 0, new_lb = 0, new_ub = 0;
        Sparse1 colj, colk, row;
        std::vector<ForcedCol> forced;
    };

    const LpProblem &orig_;
    PresolveOptions opt_;
    Status status_ = Status::NotSolved;
    LpProblem reduced_;
    std::vector<Op> ops_;
    std::vector<int> row_map_, col_map_; // reduced index -> original index

    // ---- dynamic matrix -----------------------------------------------------
    struct Ent {
        int r, c;
        double v;
    };
    std::vector<Ent> ents_;
    std::vector<char> alive_;
    std::vector<std::vector<int>> rowE_, colE_;
    std::vector<int> rcnt_, ccnt_;
    // bounds, costs
    std::vector<double> rlb_, rub_, clb_, cub_, cost_;
    double offset_ = 0;
    std::vector<char> row_on_, col_on_;
    std::vector<char> isint_; // respect_integrality: integer columns (else all 0)
    long coef_tightened_ = 0;
    std::vector<int> rq_, cq_;
    std::vector<char> rin_, cin_;

    static double
    tol_scale(double v, double tol)
    { return tol * (1.0 + std::abs(v)); }

    void push_row(int i) { if (row_on_[i] && !rin_[i]) { rin_[i] = 1; rq_.push_back(i); } }
    void push_col(int j) { if (col_on_[j] && !cin_[j]) { cin_[j] = 1; cq_.push_back(j); } }

    void
    kill(int id)
    {
        if (!alive_[id]) return;
        alive_[id] = 0;
        const Ent &e = ents_[id];
        --rcnt_[e.r];
        --ccnt_[e.c];
        push_row(e.r);
        push_col(e.c);
    }

    // drop dead ids from a list once it is mostly dead
    void
    compact(std::vector<int> &l, int cnt)
    {
        if (static_cast<int>(l.size()) <= 2 * cnt + 8) return;
        size_t w = 0;
        for (int id : l)
            if (alive_[id]) l[w++] = id;
        l.resize(w);
    }

    template <typename F>
    void
    for_row(int i, F f)
    {
        compact(rowE_[i], rcnt_[i]);
        for (int id : rowE_[i])
            if (alive_[id]) f(id);
    }
    template <typename F>
    void
    for_col(int j, F f)
    {
        compact(colE_[j], ccnt_[j]);
        for (int id : colE_[j])
            if (alive_[id]) f(id);
    }

    int
    find(int r, int c)
    {
        if (rowE_[r].size() <= colE_[c].size()) {
            for (int id : rowE_[r])
                if (alive_[id] && ents_[id].c == c) return id;
        } else {
            for (int id : colE_[c])
                if (alive_[id] && ents_[id].r == r) return id;
        }
        return -1;
    }

    // A(r, c) += d, creating or dropping the entry as needed
    void
    add_to(int r, int c, double d)
    {
        const int id = find(r, c);
        if (id >= 0) {
            ents_[id].v += d;
            if (std::abs(ents_[id].v) <= 1e-12 * (1 + std::abs(d))) kill(id);
            else { push_row(r); push_col(c); }
            return;
        }
        if (d == 0.0) return;
        ents_.push_back({r, c, d});
        alive_.push_back(1);
        const int nid = static_cast<int>(ents_.size()) - 1;
        rowE_[r].push_back(nid);
        colE_[c].push_back(nid);
        ++rcnt_[r];
        ++ccnt_[c];
        push_row(r);
        push_col(c);
    }

    void
    remove_row(int i)
    {
        row_on_[i] = 0;
        for (int id : rowE_[i]) kill(id);
        rowE_[i].clear();
    }
    void
    remove_col(int j)
    {
        col_on_[j] = 0;
        for (int id : colE_[j]) kill(id);
        colE_[j].clear();
    }

    Sparse1
    col_except(int j, int row)
    {
        Sparse1 s;
        for_col(j, [&](int id) { if (ents_[id].r != row) s.emplace_back(ents_[id].r, ents_[id].v); });
        return s;
    }
    Sparse1
    row_except(int i, int col)
    {
        Sparse1 s;
        for_row(i, [&](int id) { if (ents_[id].c != col) s.emplace_back(ents_[id].c, ents_[id].v); });
        return s;
    }

    // x_j = v: moved into the row bounds and the objective
    void
    fix_col(int j, double v)
    {
        Op op;
        op.kind = Op::FixedCol;
        op.j = j;
        op.value = v;
        ops_.push_back(op);
        offset_ += cost_[j] * v;
        for_col(j, [&](int id) {
            const int i = ents_[id].r;
            const double d = ents_[id].v * v;
            if (std::isfinite(rlb_[i])) rlb_[i] -= d;
            if (std::isfinite(rub_[i])) rub_[i] -= d;
        });
        remove_col(j);
    }

    // Activity bounds of row i (sum of a_ij x_j over the column bounds);
    // ninf_* count the infinite contributions.
    struct Act {
        double lo = 0, hi = 0;
        int ninf_lo = 0, ninf_hi = 0;
    };
    Act
    activity(int i, int skip_col = -1)
    {
        Act a;
        for_row(i, [&](int id) {
            const int j = ents_[id].c;
            if (j == skip_col) return;
            const double v = ents_[id].v;
            const double lo = v > 0 ? clb_[j] : cub_[j], hi = v > 0 ? cub_[j] : clb_[j];
            if (std::isfinite(lo)) a.lo += v * lo; else ++a.ninf_lo;
            if (std::isfinite(hi)) a.hi += v * hi; else ++a.ninf_hi;
        });
        return a;
    }

    // ---- rules ------------------------------------------------------------------
    void
    do_row(int i)
    {
        if (!row_on_[i]) return;
        if (opt_.empty_rows && rlb_[i] == -kInf && rub_[i] == kInf) {
            Op op; op.kind = Op::EmptyRow; op.i = i;
            ops_.push_back(op);
            remove_row(i);
            return;
        }
        if (rcnt_[i] == 0) {
            if (!opt_.empty_rows) return;
            if (rlb_[i] > tol_scale(rlb_[i], opt_.feas_tol) ||
                rub_[i] < -tol_scale(rub_[i], opt_.feas_tol)) {
                status_ = Status::Infeasible;
                return;
            }
            Op op; op.kind = Op::EmptyRow; op.i = i;
            ops_.push_back(op);
            row_on_[i] = 0;
            return;
        }
        if (rcnt_[i] == 1 && opt_.singleton_rows) {
            int j = -1;
            double a = 0;
            for_row(i, [&](int id) { j = ents_[id].c; a = ents_[id].v; });
            double lo = a > 0 ? rlb_[i] / a : rub_[i] / a;
            double hi = a > 0 ? rub_[i] / a : rlb_[i] / a;
            if (isint_[j]) { // an integer column takes the bound rounded inward
                if (std::isfinite(lo)) lo = std::ceil(lo - 1e-9 * (1 + std::abs(lo)));
                if (std::isfinite(hi)) hi = std::floor(hi + 1e-9 * (1 + std::abs(hi)));
            }
            Op op; op.kind = Op::SingletonRow; op.i = i; op.j = j; op.a = a;
            op.cj = cost_[j];
            op.colj = col_except(j, i);
            if (lo > clb_[j]) { clb_[j] = lo; op.tight_lb = true; }
            if (hi < cub_[j]) { cub_[j] = hi; op.tight_ub = true; }
            if (clb_[j] > cub_[j]) {
                if (isint_[j] || clb_[j] - cub_[j] > tol_scale(cub_[j], opt_.feas_tol)) {
                    status_ = Status::Infeasible;
                    return;
                }
                clb_[j] = cub_[j] = 0.5 * (clb_[j] + cub_[j]);
            }
            ops_.push_back(op);
            remove_row(i);
            push_col(j);
            return;
        }
        if (opt_.doubleton_equations && rcnt_[i] == 2 && rlb_[i] == rub_[i] &&
            std::isfinite(rlb_[i])) {
            if (doubleton(i)) return;
        }
        if (opt_.activity_rows) activity_rule(i);
    }

    // redundant and forcing rows
    void
    activity_rule(int i)
    {
        const Act a = activity(i);
        const double tl = tol_scale(rlb_[i], opt_.feas_tol), tu = tol_scale(rub_[i], opt_.feas_tol);
        const bool lo_fin = a.ninf_lo == 0, hi_fin = a.ninf_hi == 0;
        // infeasibility
        if ((lo_fin && std::isfinite(rub_[i]) && a.lo > rub_[i] + 1e3 * tu) ||
            (hi_fin && std::isfinite(rlb_[i]) && a.hi < rlb_[i] - 1e3 * tl)) {
            status_ = Status::Infeasible;
            return;
        }
        // redundant
        const bool lb_ok = rlb_[i] == -kInf || (lo_fin && a.lo >= rlb_[i] - tl);
        const bool ub_ok = rub_[i] == kInf || (hi_fin && a.hi <= rub_[i] + tu);
        if (lb_ok && ub_ok) {
            Op op; op.kind = Op::EmptyRow; op.i = i;
            ops_.push_back(op);
            remove_row(i);
            return;
        }
        // forcing: the activity bound meets the opposite row bound
        const bool force_up = lo_fin && std::isfinite(rub_[i]) && a.lo >= rub_[i] - tu;
        const bool force_lo = hi_fin && std::isfinite(rlb_[i]) && a.hi <= rlb_[i] + tl;
        if (!force_up && !force_lo) {
            if (opt_.coef_tightening) coef_tighten(i);
            return;
        }
        Op op;
        op.kind = Op::Forcing;
        op.i = i;
        op.at_upper = force_up;
        std::vector<std::pair<int, double>> fix;
        for_row(i, [&](int id) {
            const int j = ents_[id].c;
            const double v = ents_[id].v;
            const double val = (force_up == (v > 0)) ? clb_[j] : cub_[j];
            fix.emplace_back(j, val);
            if (clb_[j] != cub_[j]) op.forced.push_back({j, v, cost_[j], col_except(j, i)});
        });
        ops_.push_back(std::move(op));
        remove_row(i);
        for (auto &f : fix) fix_col(f.first, f.second);
    }

    // Coefficient tightening (Savelsbergh) of a one-sided row  sum a_j x_j <= rhs  (a
    // >= row is handled negated): for an integer column with range one (x in {l, l+1})
    // whose row is redundant at one of its two values, the coefficient and the right-hand
    // side shrink by the same amount d. The integer-feasible set is unchanged, the LP
    // relaxation gets tighter (big-M rows). Nothing to undo in postsolve: x is the same.
    void
    coef_tighten(int i)
    {
        const bool le = std::isfinite(rub_[i]) && rlb_[i] == -kInf;
        const bool ge = std::isfinite(rlb_[i]) && rub_[i] == kInf;
        if (!le && !ge) return;
        const double s = le ? 1.0 : -1.0;
        double rhs = le ? rub_[i] : -rlb_[i];
        double maxact = 0;
        bool fin = true;
        std::vector<int> ids;
        for_row(i, [&](int id) {
            const int j = ents_[id].c;
            const double a = s * ents_[id].v;
            const double m = a > 0 ? a * cub_[j] : a * clb_[j];
            if (!std::isfinite(m)) fin = false;
            maxact += m;
            ids.push_back(id);
        });
        if (!fin || maxact <= rhs) return; // unbounded activity, or redundant (handled elsewhere)
        for (int id : ids) {
            if (!alive_[id]) continue;
            const int j = ents_[id].c;
            if (!isint_[j]) continue;
            const double l = clb_[j], u = cub_[j];
            if (!std::isfinite(l) || !std::isfinite(u) || u - l != 1.0) continue;
            const double a = s * ents_[id].v;
            double d = 0, a_new = a, rhs_new = rhs, max_new = maxact;
            if (a > 0) {           // activity at x = l is maxact - a
                d = rhs - (maxact - a);
                if (d > 1e-6 * (1 + std::abs(a)) && d < a) {
                    a_new = a - d; rhs_new = rhs - d * u; max_new = maxact - d * u;
                } else d = 0;
            } else if (a < 0) {    // activity at x = u is maxact + a
                d = rhs - (maxact + a);
                if (d > 1e-6 * (1 + std::abs(a)) && d < -a) {
                    a_new = a + d; rhs_new = rhs + d * l; max_new = maxact + d * l;
                } else d = 0;
            }
            if (d == 0) continue;
            ents_[id].v = s * a_new;
            rhs = rhs_new;
            maxact = max_new;
            if (le) rub_[i] = rhs; else rlb_[i] = -rhs;
            ++coef_tightened_;
            push_col(j);
        }
    }

    // a x_j + b x_k = rhs: substitute x_k = (rhs - a x_j) / b everywhere
    bool
    doubleton(int i)
    {
        int ids[2], n = 0;
        for_row(i, [&](int id) { if (n < 2) ids[n++] = id; });
        if (n != 2) return false;
        const Ent e0 = ents_[ids[0]], e1 = ents_[ids[1]];
        // eliminate the column with fewer entries (less fill), unless its
        // coefficient is much smaller than the other one (stability)
        bool first_k = ccnt_[e0.c] < ccnt_[e1.c] ||
                       (ccnt_[e0.c] == ccnt_[e1.c] && std::abs(e0.v) >= std::abs(e1.v));
        const double big = std::max(std::abs(e0.v), std::abs(e1.v));
        if (first_k && std::abs(e0.v) < 1e-2 * big) first_k = false;
        if (!first_k && std::abs(e1.v) < 1e-2 * big) first_k = true;
        // MIP mode: only a continuous column can be substituted out
        if (isint_[(first_k ? e0 : e1).c]) {
            if (isint_[(first_k ? e1 : e0).c]) return false; // both integer
            first_k = !first_k;
            if (std::abs((first_k ? e0 : e1).v) < 1e-2 * big) return false;
        }
        const Ent &ek = first_k ? e0 : e1, &ej = first_k ? e1 : e0;
        const int j = ej.c, k = ek.c;
        const double a = ej.v, b = ek.v, rhs = rlb_[i];
        Op op;
        op.kind = Op::Doubleton;
        op.i = i; op.j = j; op.k = k; op.a = a; op.b = b; op.rhs = rhs;
        op.cj = cost_[j]; op.ck = cost_[k];
        op.colj = col_except(j, i);
        op.colk = col_except(k, i);
        op.lb_i = clb_[j]; // x_j's own bounds before the transfer
        op.ub_i = cub_[j];
        // bounds of x_k become bounds of x_j = (rhs - b x_k) / a
        auto t = [&](double xk) { return (rhs - b * xk) / a; };
        double lo = -kInf, hi = kInf;
        const bool dec = b / a > 0; // x_j decreases with x_k
        const double from_lk = std::isfinite(clb_[k]) ? t(clb_[k]) : (dec ? kInf : -kInf);
        const double from_uk = std::isfinite(cub_[k]) ? t(cub_[k]) : (dec ? -kInf : kInf);
        lo = dec ? from_uk : from_lk;
        hi = dec ? from_lk : from_uk;
        // (an infinite bound is always loosened by a finite one: no tolerance
        // arithmetic with infinities)
        if (lo > clb_[j] && (clb_[j] == -kInf || lo > clb_[j] + tol_scale(clb_[j], 1e-12))) {
            clb_[j] = lo; op.tight_lb = true; op.new_lb = lo;
        }
        if (hi < cub_[j] && (cub_[j] == kInf || hi < cub_[j] - tol_scale(cub_[j], 1e-12))) {
            cub_[j] = hi; op.tight_ub = true; op.new_ub = hi;
        }
        if (isint_[j]) { // an integer column keeps integral bounds
            if (std::isfinite(clb_[j])) clb_[j] = std::ceil(clb_[j] - 1e-9 * (1 + std::abs(clb_[j])));
            if (std::isfinite(cub_[j])) cub_[j] = std::floor(cub_[j] + 1e-9 * (1 + std::abs(cub_[j])));
        }
        if (clb_[j] > cub_[j]) {
            if (isint_[j] || clb_[j] - cub_[j] > tol_scale(cub_[j], opt_.feas_tol)) {
                status_ = Status::Infeasible;
                return true;
            }
            clb_[j] = cub_[j] = 0.5 * (clb_[j] + cub_[j]);
        }
        // substitute into the other rows of column k and the objective
        for (const auto &e : op.colk) {
            const int r = e.first;
            const double ark = e.second;
            add_to(r, j, -ark * a / b);
            const double shift = ark * rhs / b;
            if (std::isfinite(rlb_[r])) rlb_[r] -= shift;
            if (std::isfinite(rub_[r])) rub_[r] -= shift;
        }
        cost_[j] -= cost_[k] * a / b;
        offset_ += cost_[k] * rhs / b;
        ops_.push_back(std::move(op));
        remove_col(k);
        remove_row(i);
        push_col(j);
        return true;
    }

    void
    do_col(int j)
    {
        if (!col_on_[j]) return;
        if (ccnt_[j] == 0) {
            if (!opt_.empty_cols) return;
            double v;
            if (cost_[j] > 0) v = clb_[j];
            else if (cost_[j] < 0) v = cub_[j];
            else v = std::min(std::max(0.0, clb_[j]), cub_[j]);
            if (std::isinf(v)) {
                if (cost_[j] != 0) { status_ = Status::Unbounded; return; }
                v = std::isinf(clb_[j]) ? cub_[j] : clb_[j];
                if (std::isinf(v)) v = 0;
            }
            Op op; op.kind = Op::EmptyCol; op.j = j; op.value = v;
            ops_.push_back(op);
            offset_ += cost_[j] * v;
            col_on_[j] = 0;
            return;
        }
        if (clb_[j] == cub_[j]) {
            if (opt_.fixed_cols) fix_col(j, clb_[j]);
            return;
        }
        if (opt_.dual_fixing) {
            // down-lock: a row that decreasing x_j could violate (up-lock likewise)
            bool down = false, up = false;
            for_col(j, [&](int id) {
                const int i = ents_[id].r;
                const double v = ents_[id].v;
                const bool lbf = std::isfinite(rlb_[i]), ubf = std::isfinite(rub_[i]);
                if (v > 0) { down |= lbf; up |= ubf; }
                else { down |= ubf; up |= lbf; }
            });
            if (cost_[j] >= 0 && !down && std::isfinite(clb_[j])) { fix_col(j, clb_[j]); return; }
            if (cost_[j] <= 0 && !up && std::isfinite(cub_[j])) { fix_col(j, cub_[j]); return; }
        }
        if (opt_.singleton_cols && ccnt_[j] == 1) { singleton_col(j); return; }
        if (opt_.aggregate && ccnt_[j] >= 2 && ccnt_[j] <= 12) aggregate(j);
    }

    // Implied-free column in an equation: substitute
    //   x_j = (rhs - sum_{k != j} a_ik x_k) / a_ij
    // into the other rows of column j (fill limited) and the objective, then
    // drop row i and column j. With a single entry this is the classic
    // implied-free column singleton.
    bool
    aggregate(int j)
    {
        if (isint_[j]) return false; // only a continuous column can be substituted out
        constexpr int kMaxFill = 64;
        int best_i = -1, best_len = 1 << 30;
        double best_a = 0;
        double cmax = 0;
        for_col(j, [&](int id) { cmax = std::max(cmax, std::abs(ents_[id].v)); });
        for_col(j, [&](int id) {
            const int i = ents_[id].r;
            const double a = ents_[id].v;
            if (rlb_[i] != rub_[i] || !std::isfinite(rlb_[i])) return;
            if (std::abs(a) < 1e-2 * cmax) return; // stability of the substitution
            if ((ccnt_[j] - 1) * (rcnt_[i] - 1) > kMaxFill) return;
            if (rcnt_[i] < best_len) { best_len = rcnt_[i]; best_i = i; best_a = a; }
        });
        if (best_i < 0) return false;
        const int i = best_i;
        const double a = best_a, rhs = rlb_[i];
        double rmax = 0;
        for_row(i, [&](int id) { rmax = std::max(rmax, std::abs(ents_[id].v)); });
        if (std::abs(a) < 1e-2 * rmax) return false;
        // implied free: the row and the other columns' bounds keep x_j inside
        // its own bounds, so they never bind
        const Act o = activity(i, j);
        double lo = -kInf, hi = kInf;
        if (a > 0) {
            if (o.ninf_hi == 0) lo = (rhs - o.hi) / a;
            if (o.ninf_lo == 0) hi = (rhs - o.lo) / a;
        } else {
            if (o.ninf_lo == 0) lo = (rhs - o.lo) / a;
            if (o.ninf_hi == 0) hi = (rhs - o.hi) / a;
        }
        const bool free_lo = clb_[j] == -kInf || lo >= clb_[j] - tol_scale(clb_[j], opt_.feas_tol);
        const bool free_hi = cub_[j] == kInf || hi <= cub_[j] + tol_scale(cub_[j], opt_.feas_tol);
        if (!(free_lo && free_hi)) return false;
        Op op;
        op.kind = Op::ImpliedFreeSingleton;
        op.i = i; op.j = j; op.a = a; op.rhs = rhs; op.cj = cost_[j];
        op.row = row_except(i, j);
        op.colj = col_except(j, i);
        // other rows: row_r -= (a_rj / a) row_i
        for (const auto &e : op.colj) {
            const int r = e.first;
            const double f = e.second / a;
            for (const auto &rk : op.row) add_to(r, rk.first, -f * rk.second);
            if (std::isfinite(rlb_[r])) rlb_[r] -= f * rhs;
            if (std::isfinite(rub_[r])) rub_[r] -= f * rhs;
        }
        const double f = cost_[j] / a;
        for (const auto &e : op.row) { cost_[e.first] -= f * e.second; push_col(e.first); }
        offset_ += f * rhs;
        ops_.push_back(std::move(op));
        remove_col(j);
        remove_row(i);
        return true;
    }

    void
    singleton_col(int j)
    {
        if (isint_[j]) return;
        int i = -1;
        double a = 0;
        for_col(j, [&](int id) { i = ents_[id].r; a = ents_[id].v; });
        if (i < 0 || !row_on_[i] || std::abs(a) < 1e-9) return;
        if (rlb_[i] == rub_[i]) {
            aggregate(j);
            return;
        }
        if (cost_[j] == 0 && clb_[j] == -kInf && cub_[j] == kInf) {
            // a free, costless column absorbs any activity of its row
            Op op;
            op.kind = Op::FreeSingletonIneq;
            op.i = i; op.j = j; op.a = a; op.lb_i = rlb_[i]; op.ub_i = rub_[i];
            op.row = row_except(i, j);
            ops_.push_back(std::move(op));
            remove_col(j);
            remove_row(i);
        }
    }

    // ---- duplicate rows -----------------------------------------------------------
    bool
    merge_duplicates()
    {
        const size_t m = orig_.rows();
        std::unordered_map<size_t, std::vector<int>> groups;
        std::vector<std::pair<int, double>> buf;
        auto normalized = [&](int i) {
            buf.clear();
            for_row(i, [&](int id) { buf.emplace_back(ents_[id].c, ents_[id].v); });
            std::sort(buf.begin(), buf.end());
        };
        for (size_t i = 0; i < m; ++i) {
            if (!row_on_[i] || rcnt_[i] < 2) continue;
            normalized(static_cast<int>(i));
            const double fv = buf[0].second;
            size_t h = 1469598103934665603ULL;
            for (auto &e : buf) {
                const long long q = std::llround(e.second / fv * 1e9);
                h = (h ^ static_cast<size_t>(e.first)) * 1099511628211ULL;
                h = (h ^ static_cast<size_t>(q)) * 1099511628211ULL;
            }
            groups[h].push_back(static_cast<int>(i));
        }
        bool changed = false;
        std::vector<std::pair<int, double>> ri, rk;
        for (auto &g : groups) {
            auto &rows = g.second;
            for (size_t x = 0; x + 1 < rows.size(); ++x) {
                const int i = rows[x];
                if (!row_on_[i]) continue;
                normalized(i);
                ri = buf;
                for (size_t yv = x + 1; yv < rows.size(); ++yv) {
                    const int k = rows[yv];
                    if (!row_on_[k]) continue;
                    normalized(k);
                    rk = buf;
                    if (ri.size() != rk.size()) continue;
                    double alpha = 0;
                    bool par = true;
                    for (size_t t = 0; t < ri.size() && par; ++t) {
                        if (ri[t].first != rk[t].first) { par = false; break; }
                        const double rr = rk[t].second / ri[t].second;
                        if (t == 0) alpha = rr;
                        else if (std::abs(rr - alpha) > 1e-12 * std::abs(alpha)) par = false;
                    }
                    if (!par) continue;
                    // a_k = alpha a_i  =>  a_i x in [lb_k/alpha, ub_k/alpha]
                    const double ilo = alpha > 0 ? rlb_[k] / alpha : rub_[k] / alpha;
                    const double ihi = alpha > 0 ? rub_[k] / alpha : rlb_[k] / alpha;
                    Op op; op.kind = Op::DuplicateRow; op.i = i; op.k = k;
                    op.alpha = alpha; op.lb_i = rlb_[i]; op.ub_i = rub_[i];
                    op.impl_lb_k = ilo; op.impl_ub_k = ihi;
                    double nlb = std::max(rlb_[i], ilo), nub = std::min(rub_[i], ihi);
                    if (nlb > nub) {
                        if (nlb - nub > tol_scale(nub, opt_.feas_tol)) {
                            status_ = Status::Infeasible;
                            return true;
                        }
                        nlb = nub = 0.5 * (nlb + nub);
                    }
                    rlb_[i] = nlb;
                    rub_[i] = nub;
                    ops_.push_back(op);
                    remove_row(k);
                    push_row(i);
                    changed = true;
                }
            }
        }
        return changed;
    }

    // ---- parallel columns ------------------------------------------------------
    // Columns j, k with a_k = alpha a_j and c_k = alpha c_j are interchangeable up
    // to scale: x_j + alpha x_k is one variable w with the summed bounds. The
    // merged LP is equivalent; postsolve only splits w (the duals are unchanged).
    bool
    merge_parallel_cols()
    {
        const size_t n = orig_.cols();
        std::unordered_map<size_t, std::vector<int>> groups;
        std::vector<std::pair<int, double>> cj, ck;
        auto normalized = [&](int j, std::vector<std::pair<int, double>> &out) {
            out.clear();
            for_col(j, [&](int id) { out.emplace_back(ents_[id].r, ents_[id].v); });
            std::sort(out.begin(), out.end());
        };
        for (size_t j = 0; j < n; ++j) {
            if (!col_on_[j] || ccnt_[j] < 1 || isint_[j]) continue; // integer columns are never merged
            normalized(static_cast<int>(j), cj);
            const double fv = cj[0].second;
            size_t h = 1469598103934665603ULL;
            for (auto &e : cj) {
                const long long q = std::llround(e.second / fv * 1e9);
                h = (h ^ static_cast<size_t>(e.first)) * 1099511628211ULL;
                h = (h ^ static_cast<size_t>(q)) * 1099511628211ULL;
            }
            groups[h].push_back(static_cast<int>(j));
        }
        auto lo_of = [](double a, double l, double u) { return a > 0 ? a * l : a * u; };
        auto hi_of = [](double a, double l, double u) { return a > 0 ? a * u : a * l; };
        bool changed = false;
        for (auto &g : groups) {
            auto &cols = g.second;
            for (size_t x = 0; x + 1 < cols.size(); ++x) {
                const int j = cols[x];
                if (!col_on_[j]) continue;
                normalized(j, cj);
                for (size_t y2 = x + 1; y2 < cols.size(); ++y2) {
                    const int k = cols[y2];
                    if (!col_on_[k]) continue;
                    normalized(k, ck);
                    if (cj.size() != ck.size()) continue;
                    double alpha = 0;
                    bool par = true;
                    for (size_t t = 0; t < cj.size() && par; ++t) {
                        if (cj[t].first != ck[t].first) { par = false; break; }
                        const double r = ck[t].second / cj[t].second;
                        if (t == 0) alpha = r;
                        else if (std::abs(r - alpha) > 1e-12 * std::abs(alpha)) par = false;
                    }
                    if (!par) continue;
                    // costs must be proportional too
                    if (std::abs(cost_[k] - alpha * cost_[j]) > 1e-12 * (1 + std::abs(cost_[k]))) continue;
                    Op op;
                    op.kind = Op::ParallelCol;
                    op.j = j; op.k = k; op.alpha = alpha;
                    op.lb_i = clb_[j]; op.ub_i = cub_[j];
                    op.impl_lb_k = lo_of(alpha, clb_[k], cub_[k]);
                    op.impl_ub_k = hi_of(alpha, clb_[k], cub_[k]);
                    // merged bounds (an infinite end stays infinite)
                    const double nlo = op.lb_i + op.impl_lb_k, nhi = op.ub_i + op.impl_ub_k;
                    clb_[j] = std::isfinite(nlo) ? nlo : -kInf;
                    cub_[j] = std::isfinite(nhi) ? nhi : kInf;
                    ops_.push_back(op);
                    remove_col(k);
                    push_col(j);
                    changed = true;
                }
            }
        }
        return changed;
    }

    void
    run()
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        rlb_ = orig_.row_lb; rub_ = orig_.row_ub;
        clb_ = orig_.col_lb; cub_ = orig_.col_ub;
        cost_ = orig_.c;
        offset_ = orig_.offset;
        isint_.assign(n, 0);
        if (opt_.respect_integrality)
            for (size_t j = 0; j < n && j < orig_.is_integer.size(); ++j) {
                if (!orig_.is_integer[j]) continue;
                isint_[j] = 1;
                if (std::isfinite(clb_[j])) clb_[j] = std::ceil(clb_[j] - 1e-9 * (1 + std::abs(clb_[j])));
                if (std::isfinite(cub_[j])) cub_[j] = std::floor(cub_[j] + 1e-9 * (1 + std::abs(cub_[j])));
                if (clb_[j] > cub_[j]) { status_ = Status::Infeasible; return; }
            }
        row_on_.assign(m, 1);
        col_on_.assign(n, 1);
        rcnt_.assign(m, 0);
        ccnt_.assign(n, 0);
        rowE_.assign(m, {});
        colE_.assign(n, {});
        const auto *rp = orig_.A.row_ptr();
        const auto *ci = orig_.A.col_ind();
        const double *va = orig_.A.values();
        ents_.reserve(orig_.A.nnz() + orig_.A.nnz() / 4);
        for (size_t i = 0; i < m; ++i)
            for (int k = rp[i]; k < rp[i + 1]; ++k) {
                if (va[k] == 0.0) continue;
                ents_.push_back({static_cast<int>(i), ci[k], va[k]});
                alive_.push_back(1);
                const int id = static_cast<int>(ents_.size()) - 1;
                rowE_[i].push_back(id);
                colE_[ci[k]].push_back(id);
                ++rcnt_[i];
                ++ccnt_[ci[k]];
            }
        rin_.assign(m, 0);
        cin_.assign(n, 0);
        for (size_t i = 0; i < m; ++i) push_row(static_cast<int>(i));
        for (size_t j = 0; j < n; ++j) push_col(static_cast<int>(j));

        for (int pass = 0; pass < 50 && status_ == Status::NotSolved; ++pass) {
            while (status_ == Status::NotSolved && (!rq_.empty() || !cq_.empty())) {
                if (!rq_.empty()) {
                    const int i = rq_.back();
                    rq_.pop_back();
                    rin_[i] = 0;
                    do_row(i);
                } else {
                    const int j = cq_.back();
                    cq_.pop_back();
                    cin_[j] = 0;
                    do_col(j);
                }
            }
            if (status_ != Status::NotSolved) break;
            bool changed = false;
            if (opt_.duplicate_rows && merge_duplicates()) changed = true;
            if (status_ == Status::NotSolved && opt_.parallel_cols && merge_parallel_cols())
                changed = true;
            if (!changed) break;
        }
        build_reduced();
    }

    void
    build_reduced()
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        std::vector<int> new_row(m, -1), new_col(n, -1);
        for (size_t i = 0; i < m; ++i)
            if (row_on_[i]) { new_row[i] = static_cast<int>(row_map_.size()); row_map_.push_back(static_cast<int>(i)); }
        for (size_t j = 0; j < n; ++j)
            if (col_on_[j]) { new_col[j] = static_cast<int>(col_map_.size()); col_map_.push_back(static_cast<int>(j)); }
        Sparse::CooBuilder<double> b(row_map_.size(), col_map_.size());
        for (size_t id = 0; id < ents_.size(); ++id) {
            if (!alive_[id]) continue;
            const Ent &e = ents_[id];
            if (row_on_[e.r] && col_on_[e.c]) b.add(new_row[e.r], new_col[e.c], e.v);
        }
        reduced_.name = orig_.name + "_presolved";
        reduced_.A = b.build();
        for (int i : row_map_) { reduced_.row_lb.push_back(rlb_[i]); reduced_.row_ub.push_back(rub_[i]); }
        for (int j : col_map_) {
            reduced_.c.push_back(cost_[j]);
            reduced_.col_lb.push_back(clb_[j]);
            reduced_.col_ub.push_back(cub_[j]);
        }
        reduced_.offset = offset_;
        reduced_.maximize = orig_.maximize;
        if (opt_.respect_integrality) {
            reduced_.is_integer.assign(col_map_.size(), 0);
            for (size_t k = 0; k < col_map_.size(); ++k) reduced_.is_integer[k] = isint_[col_map_[k]];
        }
    }
};

} // namespace Solver
} // namespace Panini
