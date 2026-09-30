// SPDX-License-Identifier: BSD-3-Clause
//
// Branch and bound for mixed-integer LPs (minimization; a maximization problem
// is stored negated in LpProblem, see model.h).
//
//   * node LPs: DualSimplex::prepare() once, resolve() per node with the node's
//     column bounds, warm-started from the parent's final basis (the root basis
//     when memory for stored bases runs out)
//   * search: best-first with plunging (after branching, the preferred child is
//     processed next and its sibling queued)
//   * branching: pseudocosts (most fractional until costs are known)
//   * bound propagation on the rows of changed integer variables, at every node
//   * pruning: incumbent cutoff (improved by 1 when the objective is integral),
//     reduced-cost fixing
//   * heuristics: rounding and a fractional dive at the root, dives again every
//     few hundred nodes
//   * every incumbent is checked against the ORIGINAL rows, bounds and
//     integrality before it is accepted
//
// Not included (yet): MIP presolve (the LP presolve has rules that are invalid
// for integer columns), cuts, restarts, parallel search.
#pragma once

#include "solver/lp/simplex.h"
#include "solver/model.h"
#include "solver/presolve/presolve.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Panini {
namespace Solver {

enum class MilpStatus {
    Optimal,    // proven optimal within the gap tolerances
    Feasible,   // a limit was reached with an incumbent
    Infeasible,
    Unbounded,  // the LP relaxation is unbounded (or dual infeasible)
    NoSolution, // a limit was reached without an incumbent
    Error,      // the search lost nodes to numerical failures and found nothing
};

inline const char *
to_string(MilpStatus s)
{
    switch (s) {
    case MilpStatus::Optimal: return "optimal";
    case MilpStatus::Feasible: return "feasible";
    case MilpStatus::Infeasible: return "infeasible";
    case MilpStatus::Unbounded: return "unbounded";
    case MilpStatus::NoSolution: return "no solution";
    case MilpStatus::Error: return "error";
    }
    return "?";
}

struct MilpOptions {
    double time_limit = 1e100; // seconds
    long node_limit = 100000000;
    double mip_gap = 1e-4;     // relative: (incumbent - bound) / max(|incumbent|, 1e-10)
    double abs_gap = 1e-6;
    double int_tol = 1e-6;     // integrality tolerance
    double feas_tol = 1e-6;    // row feasibility tolerance (relative to 1 + |bound|)
    bool propagate = true;
    bool diving = true;
    bool verbose = false;
    bool presolve = true;      // MIP-safe presolve and coefficient tightening
    bool cuts = true;          // root cutting planes
    bool gomory = true;        // ... including Gomory mixed-integer cuts from the tableau
    int cut_rounds = 20;
    SolverOptions lp;         // options of the node LPs (scaling etc.)
};

struct MilpSolution {
    MilpStatus status = MilpStatus::NoSolution;
    std::vector<double> x;
    double objective = kInf;   // of the incumbent, in the minimization sense of LpProblem
    double best_bound = -kInf;
    double gap = kInf;         // relative
    long nodes = 0;
    long lp_iterations = 0;
    int incumbents = 0;
    int cuts = 0;              // cut rows added at the root
    double seconds = 0;
    bool has_solution() const { return !x.empty(); }
};

class Milp {
  public:
    MilpSolution
    solve(const LpProblem &p, const MilpOptions &opt)
    {
        t0_ = std::chrono::steady_clock::now();
        p_ = &p;
        base_ = &p;
        o_ = &opt;
        n_ = static_cast<int>(p.cols());
        m_ = static_cast<int>(p.rows());
        m_base_ = m_;
        res_ = MilpSolution();
        std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("Milp::solve: " + bad);
        if (m_ == 0 || n_ == 0) throw std::invalid_argument("Milp::solve needs rows and columns");

        isint_.assign(n_, 0);
        int_idx_.clear();
        for (size_t j = 0; j < p.is_integer.size(); ++j)
            if (p.is_integer[j]) { isint_[j] = 1; int_idx_.push_back(static_cast<int>(j)); }
        At_ = p.A.transpose();
        // root bounds: integer columns get integral bounds
        lb0_ = p.col_lb;
        ub0_ = p.col_ub;
        for (int j : int_idx_) {
            if (std::isfinite(lb0_[j])) lb0_[j] = std::ceil(lb0_[j] - o_->int_tol);
            if (std::isfinite(ub0_[j])) ub0_[j] = std::floor(ub0_[j] + o_->int_tol);
            if (lb0_[j] > ub0_[j]) return finish(MilpStatus::Infeasible);
        }
        // an integral objective lets the cutoff improve by a whole unit
        integral_obj_ = true;
        for (int j = 0; j < n_; ++j) {
            const double c = p.c[j];
            if (c == 0.0) continue;
            if (!isint_[j] || std::abs(c - std::round(c)) > 1e-12) { integral_obj_ = false; break; }
        }
        pc_up_.assign(n_, 0.0); pc_dn_.assign(n_, 0.0);
        pc_nup_.assign(n_, 0); pc_ndn_.assign(n_, 0);

        lp_opt_ = o_->lp;
        lp_opt_.presolve = false;
        lp_opt_.verbose = false;
        lp_opt_.deterministic = true; // a tree search must be reproducible
        simplex_.prepare(p, lp_opt_);

        // ---- root ---------------------------------------------------------
        std::vector<double> lb = lb0_, ub = ub0_;
        if (o_->propagate) {
            std::vector<int> q = int_idx_;
            if (!propagate(lb, ub, q)) return finish(MilpStatus::Infeasible);
            lb0_ = lb;
            ub0_ = ub;
        }
        LpSolution r = solve_lp(lb, ub, nullptr, &root_basis_);
        if (r.status == Status::TimeLimit) return finish(MilpStatus::NoSolution);
        if (r.status == Status::Infeasible) return finish(MilpStatus::Infeasible);
        if (r.status != Status::Optimal) {
            // dual infeasible after phase 1: unbounded (or infeasible) relaxation
            return finish(r.status == Status::Unbounded ? MilpStatus::Unbounded : MilpStatus::Error);
        }
        root_obj_ = r.primal_objective;
        log_header();
        if (o_->cuts && !int_idx_.empty()) {
            r = separate_root(r, lb, ub);
            if (r.status == Status::Infeasible) return finish(MilpStatus::Infeasible);
            if (r.status != Status::Optimal) { // a cut round failed numerically: re-solve from scratch
                r = solve_lp(lb, ub, nullptr, &root_basis_);
                if (r.status == Status::Infeasible) return finish(MilpStatus::Infeasible);
                if (r.status != Status::Optimal) return finish(MilpStatus::Error);
            }
            root_obj_ = r.primal_objective;
            if (o_->verbose && res_.cuts > 0)
                std::printf("[milp] root bound after cuts: %.10g (%d cuts)\n", root_obj_, res_.cuts);
        }
        auto root_basis_ptr = std::make_shared<std::vector<VarStatus>>(root_basis_.status);
        if (handle_lp_point(r, lb, ub)) { // integral: done
            res_.best_bound = root_obj_;
            return finish(MilpStatus::Optimal);
        }
        if (o_->diving) dive(r, lb, ub, root_basis_ptr);
        log_line("root", root_obj_);

        // ---- tree ---------------------------------------------------------
        open_.clear();
        branch_children(r, lb, ub, {}, root_obj_, 0, root_basis_ptr);
        std::unique_ptr<Node> next;
        if (!pending_.empty()) {
            next = std::move(pending_[0]);
            for (size_t k = 1; k < pending_.size(); ++k) {
                open_.push_back(std::move(pending_[k]));
                std::push_heap(open_.begin(), open_.end(), node_cmp);
            }
            pending_.clear();
        }
        while (true) {
            if (elapsed() > o_->time_limit || res_.nodes >= o_->node_limit) break;
            std::unique_ptr<Node> cur;
            if (next) cur = std::move(next);
            else if (!open_.empty()) {
                std::pop_heap(open_.begin(), open_.end(), node_cmp);
                cur = std::move(open_.back());
                open_.pop_back();
            } else break;
            if (cur->bound > cutoff() + 1e-9) continue; // pruned by the incumbent
            ++res_.nodes;
            if (!process(*cur)) continue;
            // children of cur are in pending_: plunge into the preferred one
            if (!pending_.empty()) {
                next = std::move(pending_[0]);
                for (size_t k = 1; k < pending_.size(); ++k) {
                    open_.push_back(std::move(pending_[k]));
                    std::push_heap(open_.begin(), open_.end(), node_cmp);
                }
                pending_.clear();
            }
            if (update_bound_and_check_gap(next.get())) { gap_closed_ = true; break; }
            if (o_->verbose && res_.nodes % 500 == 0) log_line("", global_bound(next.get()));
        }
        const double bound = global_bound(next.get());
        const bool exhausted = !next && open_.empty();
        res_.best_bound = exhausted && has_incumbent_ ? res_.objective : bound;
        if (exhausted || gap_closed_) {
            if (has_incumbent_) return finish(incomplete_ && !gap_closed_ ? MilpStatus::Feasible : MilpStatus::Optimal);
            return finish(incomplete_ ? MilpStatus::Error : MilpStatus::Infeasible);
        }
        return finish(has_incumbent_ ? MilpStatus::Feasible : MilpStatus::NoSolution);
    }

  private:
    struct Change { int j; double lb, ub; };
    struct Node {
        double bound = -kInf; // parent's LP objective: a valid lower bound
        int depth = 0;
        std::vector<Change> chg;
        std::shared_ptr<std::vector<VarStatus>> basis;
        int bvar = -1;        // branching variable that created the node
        bool up = false;
        double dist = 0;      // how far the parent's LP value was moved by the branch
    };
    static bool
    node_cmp(const std::unique_ptr<Node> &a, const std::unique_ptr<Node> &b)
    {
        // std heap functions build max-heaps: "less" = worse node
        if (a->bound != b->bound) return a->bound > b->bound;
        return a->depth < b->depth;
    }

    // ---- problem data ---------------------------------------------------------
    const LpProblem *p_ = nullptr;    // the LP being solved (base problem plus cut rows)
    const LpProblem *base_ = nullptr; // the model as given: what incumbents are checked against
    LpProblem pcut_;                  // base problem with cuts appended (p_ points here after cuts)
    const MilpOptions *o_ = nullptr;
    int n_ = 0, m_ = 0, m_base_ = 0; // m_base_: rows of the model as given (cuts come after)
    std::vector<char> isint_;
    std::vector<int> int_idx_;
    HostMatrix At_;
    std::vector<double> lb0_, ub0_;
    bool integral_obj_ = false;
    SolverOptions lp_opt_;
    DualSimplex simplex_;
    SimplexBasis root_basis_;
    double root_obj_ = 0;
    // pseudocosts
    std::vector<double> pc_up_, pc_dn_;
    std::vector<int> pc_nup_, pc_ndn_;
    // search state
    std::vector<std::unique_ptr<Node>> open_, pending_;
    bool has_incumbent_ = false, incomplete_ = false, gap_closed_ = false;
    MilpSolution res_;
    std::chrono::steady_clock::time_point t0_;
    std::chrono::steady_clock::time_point last_log_ = std::chrono::steady_clock::now();

    double
    elapsed() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }

    MilpSolution
    finish(MilpStatus st)
    {
        res_.status = st;
        res_.seconds = elapsed();
        if (has_incumbent_)
            res_.gap = std::max(0.0, res_.objective - res_.best_bound) /
                       std::max(std::abs(res_.objective), 1e-10);
        if (o_->verbose)
            std::printf("[milp] %s: obj %.10g bound %.10g gap %.2e nodes %ld lp-its %ld time %.2fs\n",
                to_string(st), res_.objective, res_.best_bound, res_.gap, res_.nodes,
                res_.lp_iterations, res_.seconds);
        return res_;
    }

    double
    cutoff() const
    {
        if (!has_incumbent_) return kInf;
        const double tol = std::max(o_->abs_gap, 1e-9 * std::abs(res_.objective));
        return integral_obj_ ? res_.objective - 1.0 + 1e-6 : res_.objective - tol;
    }

    double
    global_bound(const Node *next) const
    {
        double b = kInf;
        if (next) b = std::min(b, next->bound);
        if (!open_.empty()) b = std::min(b, open_.front()->bound);
        return b == kInf ? (has_incumbent_ ? res_.objective : root_obj_) : b;
    }

    bool
    update_bound_and_check_gap(const Node *next)
    {
        if (!has_incumbent_) return false;
        const double b = global_bound(next);
        res_.best_bound = b;
        const double gap = res_.objective - b;
        return gap <= o_->abs_gap || gap <= o_->mip_gap * std::max(std::abs(res_.objective), 1e-10);
    }

    void
    log_header() const
    {
        if (o_->verbose) std::printf("[milp] %d rows, %d cols, %zu integer; root LP %.10g\n", m_, n_, int_idx_.size(), root_obj_);
    }
    void
    log_line(const char *tag, double bound)
    {
        if (!o_->verbose) return;
        std::printf("[milp] %-5s nodes %8ld open %7zu bound %14.8g incumbent %14.8g  %.1fs\n", tag,
            res_.nodes, open_.size(), bound, has_incumbent_ ? res_.objective : kInf, elapsed());
    }

    // ---- LP ------------------------------------------------------------------
    LpSolution
    solve_lp(const std::vector<double> &lb, const std::vector<double> &ub,
        const SimplexBasis *warm, SimplexBasis *out)
    {
        lp_opt_.time_limit = std::max(0.0, o_->time_limit - elapsed());
        LpSolution s = simplex_.resolve(lb, ub, warm, out, nullptr, true);
        res_.lp_iterations += s.iterations;
        return s;
    }

    // ---- feasibility of a point on the ORIGINAL model --------------------------
    bool
    feasible(const std::vector<double> &x) const
    {
        return verify(*base_, x, o_->feas_tol, o_->int_tol); // the UNCUT model: cuts never decide feasibility
    }

  public:
    // Is x feasible for p: column bounds, integrality and rows, within tolerances?
    static bool
    verify(const LpProblem &p, const std::vector<double> &x, double feas_tol, double int_tol)
    {
        const int n = static_cast<int>(p.cols()), m = static_cast<int>(p.rows());
        if (static_cast<int>(x.size()) != n) return false;
        for (int j = 0; j < n; ++j) {
            const double l = p.col_lb[j], u = p.col_ub[j];
            if (!std::isfinite(x[j])) return false;
            if (x[j] < l - feas_tol * (1 + std::abs(l)) || x[j] > u + feas_tol * (1 + std::abs(u))) return false;
            if (j < static_cast<int>(p.is_integer.size()) && p.is_integer[j] &&
                std::abs(x[j] - std::round(x[j])) > int_tol)
                return false;
        }
        const auto *rp = p.A.row_ptr();
        const auto *ci = p.A.col_ind();
        const double *va = p.A.values();
        for (int i = 0; i < m; ++i) {
            double act = 0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) act += va[k] * x[ci[k]];
            const double l = p.row_lb[i], u = p.row_ub[i];
            if (act < l - feas_tol * (1 + std::abs(l)) || act > u + feas_tol * (1 + std::abs(u))) return false;
        }
        return true;
    }

  private:

    double
    objective_of(const std::vector<double> &x) const
    {
        double obj = p_->offset;
        for (int j = 0; j < n_; ++j) obj += p_->c[j] * x[j];
        return obj;
    }

    // Offer a candidate point: integers rounded, then checked on the original model.
    bool
    offer(std::vector<double> x)
    {
        for (int j : int_idx_) x[j] = std::round(x[j]);
        // continuous columns may sit a hair outside their bounds: clamp
        for (int j = 0; j < n_; ++j)
            x[j] = std::min(std::max(x[j], p_->col_lb[j]), p_->col_ub[j]);
        if (!feasible(x)) return false;
        const double obj = objective_of(x);
        if (has_incumbent_ && obj >= res_.objective) return false;
        res_.x = std::move(x);
        res_.objective = obj;
        has_incumbent_ = true;
        ++res_.incumbents;
        if (o_->verbose) log_line("inc", obj);
        return true;
    }

    // ---- cutting planes ----------------------------------------------------------
    struct Cut {
        std::vector<int> idx;
        std::vector<double> val;
        double rhs = 0;   // sum val x <= rhs
        double eff = 0;   // violation / norm at the LP point
    };

    // Complemented MIR cut of  sum a_j x_j <= beta  (row = (j, a_j) pairs) at the LP
    // point x, over the ROOT bounds (so the cut is globally valid). Every column is
    // substituted by its nearest bound into a nonnegative variable; continuous columns
    // with a positive coefficient drop out (a relaxation), integer columns are rounded
    // (Nemhauser-Wolsey MIR), continuous columns with a negative coefficient get the
    // 1/(delta (1 - f0)) factor. Several deltas are tried; the most efficacious wins.
    bool
    cmir(const std::vector<std::pair<int, double>> &row, double beta, const std::vector<double> &x,
        Cut &best) const
    {
        struct Term { int j; double g, xs, bnd; bool integer, compl_; };
        std::vector<Term> ts;
        ts.reserve(row.size());
        bool any_int = false;
        for (const auto &e : row) {
            const int j = e.first;
            const double a = e.second;
            const double l = lb0_[j], u = ub0_[j];
            const bool fl = std::isfinite(l), fu = std::isfinite(u);
            if (!fl && !fu) return false;
            const bool use_l = fl && (!fu || x[j] - l <= u - x[j]);
            Term t;
            t.j = j;
            t.integer = isint_[j] != 0;
            any_int = any_int || t.integer;
            t.compl_ = !use_l;
            t.bnd = use_l ? l : u;
            t.g = use_l ? a : -a;              // coefficient of the nonnegative variable
            t.xs = use_l ? x[j] - l : u - x[j]; // its value at the LP point
            beta -= a * t.bnd;
            if (!t.integer && t.g > 0) continue; // positive continuous term: dropped
            ts.push_back(t);
        }
        if (!any_int) return false;
        // candidate deltas: |g| of integer columns strictly inside their range
        std::vector<double> deltas;
        for (const Term &t : ts) {
            if (!t.integer || t.xs < 1e-6) continue;
            const double range = (std::isfinite(lb0_[t.j]) && std::isfinite(ub0_[t.j])) ? ub0_[t.j] - lb0_[t.j] : kInf;
            if (range - t.xs < 1e-6) continue;
            const double d = std::abs(t.g);
            if (d > 1e-9) deltas.push_back(d);
        }
        deltas.push_back(1.0);
        std::sort(deltas.begin(), deltas.end());
        deltas.erase(std::unique(deltas.begin(), deltas.end(),
                         [](double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(a, b); }),
            deltas.end());
        if (deltas.size() > 10) deltas.resize(10);
        best.eff = 0;
        bool found = false;
        std::vector<double> coef(ts.size());
        for (double delta : deltas) {
            const double b = beta / delta;
            const double fb = std::floor(b + 1e-10);
            const double f0 = b - fb;
            if (f0 < 0.05 || f0 > 0.95) continue;
            const double sc = 1.0 / (delta * (1.0 - f0));
            double viol = -fb, rhs = fb;
            for (size_t k = 0; k < ts.size(); ++k) {
                const Term &t = ts[k];
                double F;
                if (t.integer) {
                    const double gd = t.g / delta;
                    const double fl = std::floor(gd + 1e-10);
                    F = fl + std::max(0.0, gd - fl - f0) / (1.0 - f0);
                } else {
                    F = t.g * sc; // negative
                }
                coef[k] = t.compl_ ? -F : F;
                rhs += t.compl_ ? -F * t.bnd : F * t.bnd;
                viol += F * t.xs;
            }
            double nrm = 0, cmax = 0, cmin = kInf;
            for (size_t k = 0; k < ts.size(); ++k) {
                const double c = std::abs(coef[k]);
                nrm += c * c;
                if (c > 1e-12) { cmax = std::max(cmax, c); cmin = std::min(cmin, c); }
            }
            if (nrm <= 0 || cmax / cmin > 1e7) continue;
            nrm = std::sqrt(nrm);
            const double eff = viol / nrm;
            if (viol < 1e-6 * std::max(1.0, std::abs(rhs)) || eff <= best.eff) continue;
            // accept: drop negligible coefficients by relaxing the right-hand side
            Cut c;
            c.rhs = rhs;
            bool ok = true;
            for (size_t k = 0; k < ts.size(); ++k) {
                const int j = ts[k].j;
                if (std::abs(coef[k]) > 1e-9 * cmax) {
                    c.idx.push_back(j);
                    c.val.push_back(coef[k]);
                } else if (coef[k] != 0.0) {
                    const double m = coef[k] > 0 ? coef[k] * lb0_[j] : coef[k] * ub0_[j];
                    if (!std::isfinite(m)) { ok = false; break; }
                    c.rhs -= m;
                }
            }
            if (!ok || c.idx.empty()) continue;
            c.eff = eff;
            best = std::move(c);
            found = true;
        }
        return found;
    }

    // MIR cuts from every row that touches an integer column, both directions.
    void
    generate_cuts(const std::vector<double> &x, std::vector<Cut> &out) const
    {
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        std::vector<std::pair<int, double>> row;
        for (int i = 0; i < m_base_; ++i) { // the original rows only, not earlier cuts
            const int len = rp[i + 1] - rp[i];
            if (len < 2 || len > 2000) continue;
            double act = 0;
            bool has_int = false;
            for (int k = rp[i]; k < rp[i + 1]; ++k) { act += va[k] * x[ci[k]]; has_int = has_int || isint_[ci[k]]; }
            if (!has_int) continue;
            for (int dir = 0; dir < 2; ++dir) {
                const double bound = dir == 0 ? p_->row_ub[i] : p_->row_lb[i];
                if (!std::isfinite(bound)) continue;
                const double slack = dir == 0 ? bound - act : act - bound;
                if (slack > 1e-4 * (1 + std::abs(bound))) continue; // not binding
                row.clear();
                for (int k = rp[i]; k < rp[i + 1]; ++k)
                    row.emplace_back(ci[k], dir == 0 ? va[k] : -va[k]);
                Cut c;
                if (cmir(row, dir == 0 ? bound : -bound, x, c)) out.push_back(std::move(c));
            }
        }
    }

    // Gomory mixed-integer cuts from the tableau rows of fractional basic integer columns
    // (the most fractional ones). With nonbasic variables substituted by nonnegative
    // t (at lower: x - l, at upper: u - x) a row reads  x_B + sum a_j t_j = b  with
    // f0 = frac(b); the cut is  sum c_j t_j >= 1  with c_j = f_j/f0 or (1-f_j)/(1-f0)
    // for integer t_j (f_j = frac(a_j)) and a_j/f0 or -a_j/(1-f0) for continuous ones
    // (slacks count as continuous). Translated back to x (a slack is its row activity).
    void
    generate_gomory(const LpSolution &r, std::vector<Cut> &out)
    {
        simplex_.begin_tableau();
        std::vector<std::pair<double, int>> cand;
        for (int pos = 0; pos < simplex_.basis_size(); ++pos) {
            const int j = simplex_.basic_var(pos);
            if (j >= n_ || !isint_[j]) continue;
            const double f = r.x[j] - std::floor(r.x[j]);
            if (f < 0.05 || f > 0.95) continue;
            cand.emplace_back(std::abs(f - 0.5), pos);
        }
        std::sort(cand.begin(), cand.end());
        const size_t rows_max = std::min<size_t>(cand.size(), 50);
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        std::vector<double> acc(n_, 0.0);
        std::vector<char> seen(n_, 0);
        std::vector<int> touched;
        DualSimplex::TableauRow tr;
        for (size_t k = 0; k < rows_max; ++k) {
            const int pos = cand[k].second;
            const int jb = simplex_.basic_var(pos);
            if (!simplex_.tableau_row(pos, tr)) continue;
            const double f0 = r.x[jb] - std::floor(r.x[jb]);
            bool ok = true;
            double rhs = 1.0; // sum a x >= rhs
            touched.clear();
            auto add = [&](int v, double a) {
                if (!seen[v]) { seen[v] = 1; touched.push_back(v); }
                acc[v] += a;
            };
            for (size_t q = 0; q < tr.var.size() && ok; ++q) {
                const int v = tr.var[q];
                if (tr.state[q] == 2) { ok = false; break; } // a free nonbasic variable
                const bool atu = tr.state[q] == 1;
                double l, u;
                bool integer = false;
                if (v < n_) { l = lb0_[v]; u = ub0_[v]; integer = isint_[v] != 0; }
                else { l = p_->row_lb[v - n_]; u = p_->row_ub[v - n_]; }
                const double bound = atu ? u : l;
                if (!std::isfinite(bound)) { ok = false; break; }
                const double a = atu ? -tr.coef[q] : tr.coef[q]; // coefficient of t_v >= 0
                double c;
                if (integer) {
                    double fj = a - std::floor(a);
                    if (fj < 1e-9 || fj > 1 - 1e-9) fj = 0;
                    c = fj <= f0 ? fj / f0 : (1 - fj) / (1 - f0);
                } else {
                    c = a >= 0 ? a / f0 : -a / (1 - f0);
                }
                if (c == 0) continue;
                const double sgn = atu ? -1.0 : 1.0; // t = sgn (x - bound)
                rhs += c * sgn * bound;
                if (v < n_) add(v, c * sgn);
                else {
                    const int i = v - n_;
                    for (int e = rp[i]; e < rp[i + 1]; ++e) add(ci[e], c * sgn * va[e]);
                }
            }
            Cut cut;
            if (ok) {
                // sum a x >= rhs  ->  sum (-a) x <= -rhs
                double cmax = 0, cmin = kInf, nrm = 0, lhs = 0;
                for (int v : touched) {
                    const double c = std::abs(acc[v]);
                    if (c > 1e-12) { cmax = std::max(cmax, c); cmin = std::min(cmin, c); }
                }
                cut.rhs = -rhs;
                for (int v : touched) {
                    const double a = -acc[v];
                    if (std::abs(a) > 1e-9 * cmax && a != 0.0) {
                        cut.idx.push_back(v);
                        cut.val.push_back(a);
                        nrm += a * a;
                        lhs += a * r.x[v];
                    } else if (a != 0.0) { // negligible: relax the right-hand side
                        const double m = a > 0 ? a * lb0_[v] : a * ub0_[v];
                        if (!std::isfinite(m)) { ok = false; break; }
                        cut.rhs -= m;
                    }
                }
                const double viol = lhs - cut.rhs;
                if (ok && !cut.idx.empty() && cmax > 0 && cmax / cmin < 1e7 && nrm > 0 &&
                    viol > 1e-6 * std::max(1.0, std::abs(cut.rhs))) {
                    cut.eff = viol / std::sqrt(nrm);
                    if (cut.eff > 1e-5) out.push_back(std::move(cut));
                }
            }
            for (int v : touched) { acc[v] = 0; seen[v] = 0; }
        }
    }

    // Append cut rows to the LP (p_ switches to pcut_) and re-prepare the simplex.
    void
    add_cuts(const std::vector<Cut> &cuts)
    {
        LpProblem t = *p_;
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        std::vector<int32_t> nrp(rp, rp + m_ + 1), nci(ci, ci + p_->A.nnz());
        std::vector<double> nva(va, va + p_->A.nnz());
        for (const Cut &c : cuts) {
            std::vector<size_t> ord(c.idx.size());
            for (size_t k = 0; k < ord.size(); ++k) ord[k] = k;
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return c.idx[a] < c.idx[b]; });
            for (size_t k : ord) { nci.push_back(c.idx[k]); nva.push_back(c.val[k]); }
            nrp.push_back(static_cast<int32_t>(nci.size()));
            t.row_lb.push_back(-kInf);
            t.row_ub.push_back(c.rhs);
        }
        t.A = HostMatrix(static_cast<size_t>(m_) + cuts.size(), static_cast<size_t>(n_), nrp, nci, nva);
        pcut_ = std::move(t);
        p_ = &pcut_;
        m_ += static_cast<int>(cuts.size());
        At_ = pcut_.A.transpose();
        simplex_.prepare(pcut_, lp_opt_);
    }

    // Cutting-plane rounds at the root. Returns the last LP solution; `status` of the
    // returned solution is Optimal, Infeasible (the cuts proved it), or something
    // else when the LP failed (the caller then re-solves from scratch).
    LpSolution
    separate_root(LpSolution r, const std::vector<double> &lb, const std::vector<double> &ub)
    {
        double last = r.primal_objective;
        int stall = 0;
        size_t total = 0;
        const size_t cap = std::max<size_t>(300, 2 * static_cast<size_t>(m_base_));
        const double t_end = std::min(o_->time_limit, elapsed() + 0.25 * std::min(o_->time_limit, 600.0));
        for (int round = 0; round < o_->cut_rounds && elapsed() < t_end; ++round) {
            std::vector<int> fr;
            if (fractional(r.x, fr) == 0) break;
            std::vector<Cut> cand, pick;
            generate_cuts(r.x, cand);
            if (o_->gomory) generate_gomory(r, cand);
            if (cand.empty()) break;
            std::sort(cand.begin(), cand.end(), [](const Cut &a, const Cut &b) { return a.eff > b.eff; });
            const size_t per_round = std::max<size_t>(20, std::min<size_t>(200, static_cast<size_t>(m_base_) / 4 + 10));
            for (const Cut &c : cand) {
                if (pick.size() >= per_round) break;
                // parallelism filter: skip a cut nearly parallel to one already chosen
                bool par = false;
                for (const Cut &q : pick) {
                    double dot = 0, n1 = 0, n2 = 0;
                    size_t a = 0, b = 0;
                    for (double v : c.val) n1 += v * v;
                    for (double v : q.val) n2 += v * v;
                    while (a < c.idx.size() && b < q.idx.size()) {
                        if (c.idx[a] == q.idx[b]) dot += c.val[a++] * q.val[b++];
                        else if (c.idx[a] < q.idx[b]) ++a; else ++b;
                    }
                    if (dot > 0.95 * std::sqrt(n1 * n2)) { par = true; break; }
                }
                if (!par) pick.push_back(c);
            }
            if (pick.empty()) break;
            // previous basis plus a basic slack for every new row: dual feasible
            SimplexBasis warm = root_basis_;
            for (size_t k = 0; k < pick.size(); ++k) warm.status.push_back(VarStatus::Basic);
            add_cuts(pick);
            total += pick.size();
            SimplexBasis nb;
            LpSolution nr = solve_lp(lb, ub, &warm, &nb);
            if (nr.status == Status::Infeasible) return nr;
            if (nr.status != Status::Optimal) {
                nr = solve_lp(lb, ub, nullptr, &nb); // retry cold
                if (nr.status != Status::Optimal) return nr;
            }
            r = std::move(nr);
            root_basis_ = std::move(nb);
            res_.cuts = static_cast<int>(total);
            const double obj = r.primal_objective;
            if (o_->verbose)
                std::printf("[milp] cut round %2d: +%zu cuts (%zu total), bound %.10g\n", round + 1,
                    pick.size(), total, obj);
            if (obj - last < 1e-5 * (1 + std::abs(obj))) { if (++stall >= 3) break; }
            else stall = 0;
            last = obj;
            if (total > cap) break;
        }
        return purge_cuts(lb, ub, std::move(r));
    }

    // Drop the cut rows that are not binding at the root optimum (their slack is basic):
    // they only make every node LP bigger. The remaining basis stays valid.
    LpSolution
    purge_cuts(const std::vector<double> &lb, const std::vector<double> &ub, LpSolution r)
    {
        if (m_ == m_base_ || r.status != Status::Optimal) return r;
        std::vector<int> keep;
        for (int i = 0; i < m_; ++i)
            if (i < m_base_ || root_basis_.status[n_ + i] != VarStatus::Basic) keep.push_back(i);
        if (static_cast<int>(keep.size()) == m_) return r;
        const int dropped = m_ - static_cast<int>(keep.size());
        LpProblem t = *p_;
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        std::vector<int32_t> nrp{0}, nci;
        std::vector<double> nva;
        t.row_lb.clear();
        t.row_ub.clear();
        SimplexBasis warm;
        warm.status.assign(root_basis_.status.begin(), root_basis_.status.begin() + n_);
        for (int i : keep) {
            for (int e = rp[i]; e < rp[i + 1]; ++e) { nci.push_back(ci[e]); nva.push_back(va[e]); }
            nrp.push_back(static_cast<int32_t>(nci.size()));
            t.row_lb.push_back(p_->row_lb[i]);
            t.row_ub.push_back(p_->row_ub[i]);
            warm.status.push_back(root_basis_.status[n_ + i]);
        }
        t.A = HostMatrix(keep.size(), static_cast<size_t>(n_), nrp, nci, nva);
        pcut_ = std::move(t);
        p_ = &pcut_;
        m_ = static_cast<int>(keep.size());
        At_ = pcut_.A.transpose();
        simplex_.prepare(pcut_, lp_opt_);
        SimplexBasis nb;
        LpSolution nr = solve_lp(lb, ub, &warm, &nb);
        if (nr.status != Status::Optimal) nr = solve_lp(lb, ub, nullptr, &nb);
        if (nr.status == Status::Optimal) root_basis_ = std::move(nb);
        if (o_->verbose)
            std::printf("[milp] purged %d inactive cuts, %d kept\n", dropped, m_ - m_base_);
        res_.cuts = m_ - m_base_;
        return nr;
    }

    // ---- propagation -----------------------------------------------------------
    // Tightens integer bounds from the rows of the variables in `queue` (and of
    // every variable tightened on the way). false: a row cannot be satisfied.
    bool
    propagate(std::vector<double> &lb, std::vector<double> &ub, std::vector<int> queue) const
    {
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        const auto *cp = At_.row_ptr();
        const auto *cr = At_.col_ind();
        std::vector<int> rows_stamp(m_, -1);
        size_t work = 0;
        const size_t budget = 20 * (p_->A.nnz() + 100);
        int pass = 0;
        std::vector<int> rows;
        while (!queue.empty() && work < budget) {
            rows.clear();
            ++pass;
            for (int j : queue)
                for (int k = cp[j]; k < cp[j + 1]; ++k) {
                    const int i = cr[k];
                    if (rows_stamp[i] != pass) { rows_stamp[i] = pass; rows.push_back(i); }
                }
            queue.clear();
            for (int i : rows) {
                const double rl = p_->row_lb[i], ru = p_->row_ub[i];
                double minact = 0, maxact = 0;
                int ninf_min = 0, ninf_max = 0;
                for (int k = rp[i]; k < rp[i + 1]; ++k) {
                    const int j = ci[k];
                    const double a = va[k];
                    const double lo = a > 0 ? lb[j] : ub[j], hi = a > 0 ? ub[j] : lb[j];
                    if (std::isfinite(lo)) minact += a * lo; else ++ninf_min;
                    if (std::isfinite(hi)) maxact += a * hi; else ++ninf_max;
                }
                work += rp[i + 1] - rp[i];
                const double tol = o_->feas_tol * (1 + std::abs(rl)), tolu = o_->feas_tol * (1 + std::abs(ru));
                if ((ninf_min == 0 && std::isfinite(ru) && minact > ru + tolu) ||
                    (ninf_max == 0 && std::isfinite(rl) && maxact < rl - tol))
                    return false;
                for (int k = rp[i]; k < rp[i + 1]; ++k) {
                    const int j = ci[k];
                    if (!isint_[j]) continue;
                    const double a = va[k];
                    const double lo = a > 0 ? lb[j] : ub[j], hi = a > 0 ? ub[j] : lb[j];
                    // residual activity of the other columns
                    double nl = -kInf, nu = kInf;
                    if (std::isfinite(ru)) { // a x_j <= ru - minact(others)
                        const bool others_fin = ninf_min == 0 || (ninf_min == 1 && !std::isfinite(lo));
                        if (others_fin) {
                            const double rest = minact - (std::isfinite(lo) ? a * lo : 0.0);
                            const double bound = (ru - rest) / a;
                            if (a > 0) nu = std::min(nu, bound); else nl = std::max(nl, bound);
                        }
                    }
                    if (std::isfinite(rl)) { // a x_j >= rl - maxact(others)
                        const bool others_fin = ninf_max == 0 || (ninf_max == 1 && !std::isfinite(hi));
                        if (others_fin) {
                            const double rest = maxact - (std::isfinite(hi) ? a * hi : 0.0);
                            const double bound = (rl - rest) / a;
                            if (a > 0) nl = std::max(nl, bound); else nu = std::min(nu, bound);
                        }
                    }
                    bool changed = false;
                    if (std::isfinite(nl)) {
                        const double v = std::ceil(nl - 1e-6 * (1 + std::abs(nl)));
                        if (v > lb[j] + 0.5) { lb[j] = v; changed = true; }
                    }
                    if (std::isfinite(nu)) {
                        const double v = std::floor(nu + 1e-6 * (1 + std::abs(nu)));
                        if (v < ub[j] - 0.5) { ub[j] = v; changed = true; }
                    }
                    if (lb[j] > ub[j]) return false;
                    if (changed) queue.push_back(j);
                }
            }
        }
        return true;
    }

    // ---- nodes ---------------------------------------------------------------
    static void
    apply(std::vector<double> &lb, std::vector<double> &ub, const std::vector<Change> &chg)
    {
        for (const Change &c : chg) {
            lb[c.j] = std::max(lb[c.j], c.lb);
            ub[c.j] = std::min(ub[c.j], c.ub);
        }
    }

    // Fractional integer columns of x, with their fractional distance.
    int
    fractional(const std::vector<double> &x, std::vector<int> &out) const
    {
        out.clear();
        for (int j : int_idx_)
            if (std::abs(x[j] - std::round(x[j])) > o_->int_tol) out.push_back(j);
        return static_cast<int>(out.size());
    }

    // An LP point: integral -> incumbent (true); else try rounding. Returns whether
    // the point was integral.
    bool
    handle_lp_point(const LpSolution &s, const std::vector<double> &, const std::vector<double> &)
    {
        std::vector<int> fr;
        if (fractional(s.x, fr) == 0) {
            if (offer(s.x)) return true;
            // integral but rejected by the original-model check: treat as fractional
            // numerical noise only if it is also worse than the incumbent
            return has_incumbent_ && s.primal_objective >= res_.objective - 1e-9;
        }
        // simple rounding
        std::vector<double> xr = s.x;
        for (int j : fr) xr[j] = std::round(xr[j]);
        offer(std::move(xr));
        return false;
    }

    // Pseudocost score of branching on j from LP value x; columns without history use
    // the averages avg_up / avg_dn of the known ones.
    double
    score(int j, double x, double avg_up, double avg_dn) const
    {
        const double f = x - std::floor(x);
        const double pu = pc_nup_[j] > 0 ? pc_up_[j] / pc_nup_[j] : avg_up;
        const double pd = pc_ndn_[j] > 0 ? pc_dn_[j] / pc_ndn_[j] : avg_dn;
        const double up = std::max(pu * (1 - f), 1e-6), dn = std::max(pd * f, 1e-6);
        return up * dn;
    }

    // Two children of an LP point (branching on the best column); they go to pending_
    // with the preferred (rounding-direction) child first.
    void
    branch_children(const LpSolution &s, const std::vector<double> &, const std::vector<double> &,
        const std::vector<Change> &chg, double bound, int depth,
        const std::shared_ptr<std::vector<VarStatus>> &basis)
    {
        std::vector<int> fr;
        fractional(s.x, fr);
        if (fr.empty()) return;
        int best = fr[0];
        double best_score = -1;
        bool any_pc = false;
        double su = 0, sd = 0, avg_up = 1, avg_dn = 1;
        int cu = 0, cd = 0;
        for (int k : int_idx_) {
            if (pc_nup_[k] > 0) { su += pc_up_[k] / pc_nup_[k]; ++cu; }
            if (pc_ndn_[k] > 0) { sd += pc_dn_[k] / pc_ndn_[k]; ++cd; }
        }
        any_pc = cu + cd > 0;
        if (cu) avg_up = su / cu;
        if (cd) avg_dn = sd / cd;
        for (int j : fr) {
            const double f = std::abs(s.x[j] - std::floor(s.x[j]) - 0.5);
            const double sc = any_pc ? score(j, s.x[j], avg_up, avg_dn) : 0.5 - f; // most fractional first
            if (sc > best_score) { best_score = sc; best = j; }
        }
        const double xv = s.x[best];
        const double fl = std::floor(xv), fracp = xv - fl;
        auto make = [&](bool up) {
            auto nd = std::make_unique<Node>();
            nd->bound = bound;
            nd->depth = depth + 1;
            nd->chg = chg;
            nd->basis = basis;
            nd->bvar = best;
            nd->up = up;
            nd->dist = up ? 1 - fracp : fracp;
            nd->chg.push_back(up ? Change{best, fl + 1, kInf} : Change{best, -kInf, fl});
            return nd;
        };
        pending_.clear();
        // Child order: toward the smaller predicted objective increase when the column has
        // a pseudocost history on both sides, else toward the nearest integer.
        bool prefer_up = fracp >= 0.5;
        if (pc_nup_[best] > 0 && pc_ndn_[best] > 0) {
            const double up_cost = pc_up_[best] / pc_nup_[best] * (1 - fracp);
            const double dn_cost = pc_dn_[best] / pc_ndn_[best] * fracp;
            prefer_up = up_cost < dn_cost;
        }
        if (prefer_up) { pending_.push_back(make(true)); pending_.push_back(make(false)); }
        else { pending_.push_back(make(false)); pending_.push_back(make(true)); }
    }

    // Solve a node. Children (if any) are left in pending_. false: node is done
    // (pruned, infeasible or integral) without children.
    bool
    process(Node &nd)
    {
        pending_.clear();
        std::vector<double> lb = lb0_, ub = ub0_;
        apply(lb, ub, nd.chg);
        for (int j = 0; j < n_; ++j) if (lb[j] > ub[j]) return false;
        if (o_->propagate) {
            std::vector<int> q;
            for (const Change &c : nd.chg) q.push_back(c.j);
            if (!propagate(lb, ub, q)) return false;
        }
        SimplexBasis basis;
        SimplexBasis warm;
        const SimplexBasis *wp = nullptr;
        if (nd.basis) { warm.status = *nd.basis; wp = &warm; }
        else wp = &root_basis_;
        LpSolution s = solve_lp(lb, ub, wp, &basis);
        if (s.status != Status::Optimal && s.status != Status::Infeasible && s.status != Status::TimeLimit) {
            s = solve_lp(lb, ub, nullptr, &basis); // retry from the slack basis
        }
        if (s.status == Status::TimeLimit) { gap_closed_ = false; requeue(nd); return false; }
        if (s.status == Status::Infeasible) return false;
        if (s.status != Status::Optimal) { incomplete_ = true; return false; }
        const double obj = s.primal_objective;
        // pseudocost update
        if (nd.bvar >= 0 && nd.dist > 1e-9 && std::isfinite(nd.bound)) {
            const double gain = std::max(0.0, obj - nd.bound) / nd.dist;
            if (nd.up) { pc_up_[nd.bvar] += gain; ++pc_nup_[nd.bvar]; }
            else { pc_dn_[nd.bvar] += gain; ++pc_ndn_[nd.bvar]; }
        }
        if (obj > cutoff() + 1e-9) return false;
        if (handle_lp_point(s, lb, ub)) return false; // integral point handled
        if (obj > cutoff() + 1e-9) return false;
        // reduced-cost fixing for the subtree
        std::vector<Change> chg = nd.chg;
        if (has_incumbent_) reduced_cost_fixing(s, lb, ub, obj, chg);
        auto bptr = std::make_shared<std::vector<VarStatus>>(std::move(basis.status));
        // no incumbent yet: dive from this node's point now and then (each node's LP
        // differs, so each dive explores something new)
        if (o_->diving && !has_incumbent_ && res_.nodes % 100 == 1) dive(s, lb, ub, bptr);
        if (obj > cutoff() + 1e-9) return false;
        branch_children(s, lb, ub, chg, std::max(obj, nd.bound), nd.depth, bptr);
        return !pending_.empty();
    }

    // A node interrupted by the time limit goes back to the queue so the final
    // bound stays valid.
    void
    requeue(Node &nd)
    {
        auto nn = std::make_unique<Node>(nd);
        open_.push_back(std::move(nn));
        std::push_heap(open_.begin(), open_.end(), node_cmp);
    }

    void
    reduced_cost_fixing(const LpSolution &s, const std::vector<double> &lb,
        const std::vector<double> &ub, double obj, std::vector<Change> &chg) const
    {
        const double room = cutoff() - obj;
        if (!(room >= 0)) return;
        const auto *cp = At_.row_ptr();
        const auto *cr = At_.col_ind();
        const double *cv = At_.values();
        for (int j : int_idx_) {
            if (lb[j] == ub[j]) continue;
            double z = p_->c[j];
            for (int k = cp[j]; k < cp[j + 1]; ++k) z -= cv[k] * s.y[cr[k]];
            const double x = s.x[j];
            if (z > 1e-7 && x <= lb[j] + 1e-9 * (1 + std::abs(lb[j]))) {
                const double nu = lb[j] + std::floor(room / z + 1e-9);
                if (nu < ub[j]) chg.push_back({j, -kInf, nu});
            } else if (z < -1e-7 && x >= ub[j] - 1e-9 * (1 + std::abs(ub[j]))) {
                const double nl = ub[j] - std::floor(room / -z + 1e-9);
                if (nl > lb[j]) chg.push_back({j, nl, kInf});
            }
        }
    }

    // ---- diving --------------------------------------------------------------
    // Fractional diving from an LP point: fix the least fractional column to its
    // rounding, re-solve, flip once on infeasibility.
    void
    dive(const LpSolution &start, std::vector<double> lb, std::vector<double> ub,
        const std::shared_ptr<std::vector<VarStatus>> &basis0)
    {
        const double t_end = std::min(o_->time_limit, elapsed() + 0.1 * std::min(o_->time_limit, 600.0));
        LpSolution cur = start;
        SimplexBasis warm;
        warm.status = *basis0;
        std::vector<int> fr;
        const int max_depth = std::min<int>(400, 2 * static_cast<int>(int_idx_.size()) + 10);
        for (int depth = 0; depth < max_depth && elapsed() < t_end; ++depth) {
            if (fractional(cur.x, fr) == 0) { offer(cur.x); return; }
            // try rounding at every step (cheap)
            { std::vector<double> xr = cur.x; for (int j : fr) xr[j] = std::round(xr[j]); offer(std::move(xr)); }
            int best = fr[0];
            double bd = 1;
            for (int j : fr) {
                const double d = std::abs(cur.x[j] - std::round(cur.x[j]));
                if (d < bd) { bd = d; best = j; }
            }
            const double r = std::round(cur.x[best]);
            bool ok = false;
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                std::vector<double> l2 = lb, u2 = ub;
                const double target = attempt == 0 ? r : (r > cur.x[best] ? r - 1 : r + 1);
                if (target < l2[best] || target > u2[best]) continue;
                l2[best] = u2[best] = target;
                if (o_->propagate) { std::vector<int> q{best}; if (!propagate(l2, u2, q)) continue; }
                SimplexBasis nb;
                LpSolution s = solve_lp(l2, u2, &warm, &nb);
                if (s.status != Status::Optimal) continue;
                if (s.primal_objective > cutoff() + 1e-9) continue;
                lb = std::move(l2); ub = std::move(u2);
                cur = std::move(s);
                warm = std::move(nb);
                ok = true;
            }
            if (!ok) return;
        }
    }
};

// Branch and bound without presolve.
inline MilpSolution
solve_milp_raw(const LpProblem &p, const MilpOptions &opt)
{
    Milp m;
    return m.solve(p, opt);
}

// MIP presolve (integrality-respecting reductions and coefficient tightening),
// branch and bound on the reduced problem, postsolve, and a feasibility check of
// the result on the ORIGINAL model; if that check fails the problem is solved again
// without presolve.
inline MilpSolution
solve_milp(const LpProblem &p, const MilpOptions &opt = MilpOptions())
{
    if (!opt.presolve) return solve_milp_raw(p, opt);
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    PresolveOptions po;
    po.respect_integrality = true;
    po.coef_tightening = true;
    Presolve pre(p, po);
    MilpSolution s;
    if (pre.status() == Status::Infeasible || pre.status() == Status::Unbounded) {
        s.status = pre.status() == Status::Infeasible ? MilpStatus::Infeasible : MilpStatus::Unbounded;
        s.seconds = secs();
        return s;
    }
    const LpProblem &r = pre.reduced();
    if (opt.verbose)
        std::printf("[milp] presolve: %zux%zu -> %zux%zu, %ld coefficients tightened\n", p.rows(),
            p.cols(), r.rows(), r.cols(), pre.coefficients_tightened());
    LpSolution rs;
    rs.status = Status::Optimal;
    if (r.rows() > 0 && r.cols() > 0) {
        MilpOptions ro = opt;
        ro.time_limit = std::max(0.0, opt.time_limit - secs());
        s = solve_milp_raw(r, ro);
        if (!s.has_solution()) { s.seconds = secs(); return s; }
        rs.x = s.x;
    }
    LpSolution e = pre.postsolve(rs);
    if (Milp::verify(p, e.x, opt.feas_tol, opt.int_tol)) {
        s.x = e.x;
        double obj = p.offset;
        for (size_t j = 0; j < p.cols(); ++j) obj += p.c[j] * e.x[j];
        s.objective = obj;
        if (r.rows() == 0 || r.cols() == 0) { s.status = MilpStatus::Optimal; s.best_bound = obj; s.gap = 0; }
        s.seconds = secs();
        return s;
    }
    // the mapped-back point fails on the original model: do not trust the presolve
    if (opt.verbose) std::printf("[milp] postsolved point infeasible on the original model: solving without presolve\n");
    MilpOptions fb = opt;
    fb.time_limit = std::max(0.0, opt.time_limit - secs());
    MilpSolution f = solve_milp_raw(p, fb);
    f.seconds = secs();
    return f;
}

} // namespace Solver
} // namespace Panini
