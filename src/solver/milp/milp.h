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
    SolverOptions lp;          // options of the node LPs (scaling etc.)
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
        o_ = &opt;
        n_ = static_cast<int>(p.cols());
        m_ = static_cast<int>(p.rows());
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
    const LpProblem *p_ = nullptr;
    const MilpOptions *o_ = nullptr;
    int n_ = 0, m_ = 0;
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
        const double tol = o_->feas_tol;
        for (int j = 0; j < n_; ++j) {
            const double l = p_->col_lb[j], u = p_->col_ub[j];
            if (x[j] < l - tol * (1 + std::abs(l)) || x[j] > u + tol * (1 + std::abs(u))) return false;
            if (isint_[j] && std::abs(x[j] - std::round(x[j])) > o_->int_tol) return false;
        }
        const auto *rp = p_->A.row_ptr();
        const auto *ci = p_->A.col_ind();
        const double *va = p_->A.values();
        for (int i = 0; i < m_; ++i) {
            double act = 0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) act += va[k] * x[ci[k]];
            const double l = p_->row_lb[i], u = p_->row_ub[i];
            if (act < l - tol * (1 + std::abs(l)) || act > u + tol * (1 + std::abs(u))) return false;
        }
        return true;
    }

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
        if (fracp >= 0.5) { pending_.push_back(make(true)); pending_.push_back(make(false)); }
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

// Convenience wrapper.
inline MilpSolution
solve_milp(const LpProblem &p, const MilpOptions &opt = MilpOptions())
{
    Milp m;
    return m.solve(p, opt);
}

} // namespace Solver
} // namespace Panini
