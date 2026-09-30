// SPDX-License-Identifier: BSD-3-Clause
//
// Bounded-variable revised dual simplex for LP.
//
//   min c^T x   s.t.  A x - w = 0,  lb <= x <= ub,  l <= w <= u
//
// Variables 0..n-1 are the columns, n..n+m-1 the row slacks w (column -e_i).
// The basis is m variables; the initial basis is all slacks. Ingredients:
//   - sparse LU of the basis with product-form updates (basis_lu.h)
//   - dual steepest-edge pricing
//   - bound-flipping (long-step) ratio test with Harris tolerances
//   - dual phase 1 by artificial bounds (free -> [-1000,1000], lower-only ->
//     [0,1], upper-only -> [-1,0], boxed/fixed -> [0,0])
//   - cost perturbation when the iteration stalls on dual degeneracy, and cost
//     shifting for small dual infeasibilities
// A basis (nonbasic-at-bound flags) can be returned and passed back as a warm
// start, which branch and bound and crossover use.
//
// The problem is scaled internally (Ruiz + Pock-Chambolle) and the solution
// is verified on the original problem by evaluate_solution().
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "solver/lp/basis_lu.h"
#include "solver/model.h"
#include "solver/scaling.h"
#include "sparse/sparse.h"
#include <chrono>
#include <cstdio>
#include <random>

namespace Panini {
namespace Solver {

enum class VarStatus : int8_t { Basic = 0, AtLower = 1, AtUpper = 2, AtZero = 3 };

struct SimplexBasis {
    std::vector<VarStatus> status; // size n + m (columns then row slacks)
    bool valid() const { return !status.empty(); }
};

// Decides when to refactor the basis: the average cost per iteration over a
// refactorization cycle, (T_factor + sum of iteration times) / k, is minimal
// where the current iteration time crosses that average (iterations get slower
// as the eta file grows). Iteration times are smoothed over a few iterations.
struct RefactorPolicy {
    using clock = std::chrono::steady_clock;
    double t_factor = 0, sum = 0, smooth = 0;
    // > 1 refactors later: large hypersparse bases have noisy iteration times that would
    // otherwise trigger early, and their refactorization is expensive (46% of mcf50k)
    double margin = 1.0;
    int fixed_freq = 0; // > 0: refactor every fixed_freq iterations (reproducible runs)
    int k = 0;
    clock::time_point t0 = clock::now();

    void begin_cycle(double factor_seconds) { t_factor = factor_seconds; sum = 0; smooth = 0; k = 0; t0 = clock::now(); }
    // call once per iteration; returns true when a refactorization pays off
    bool
    tick(int max_updates)
    {
        const auto t1 = clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        ++k;
        sum += dt;
        smooth = k == 1 ? dt : 0.8 * smooth + 0.2 * dt;
        if (fixed_freq > 0) return k >= fixed_freq; // deterministic: no wall-clock dependence
        if (k >= max_updates) return true;
        if (k < 15) return false;
        return smooth > margin * (t_factor + sum) / k;
    }
};

class DualSimplex {
  public:
    // warm: optional starting basis; out: receives the final basis.
    LpSolution
    solve(const LpProblem &p, const SolverOptions &opt,
        const SimplexBasis *warm = nullptr, SimplexBasis *out = nullptr,
        const LpSolution *start = nullptr)
    {
        prepare(p, opt);
        return resolve(p.col_lb, p.col_ub, warm, out, start, false);
    }

    // One-time setup for a problem: scaling, transposes, slack columns. `p` and
    // `opt` must outlive every resolve() call. Together with resolve() this is the
    // node-LP interface of branch and bound (no matrix copy per node).
    void
    prepare(const LpProblem &p, const SolverOptions &opt)
    {
        t_start_ = std::chrono::steady_clock::now();
        opt_ = &opt;
        orig_ = &p;
        n_ = static_cast<int>(p.cols());
        m_ = static_cast<int>(p.rows());
        N_ = n_ + m_;
        if (m_ == 0 || n_ == 0)
            throw std::invalid_argument("DualSimplex::solve needs rows and columns");
        std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("DualSimplex::solve: " + bad);

        // ---- scaling ------------------------------------------------------
        if (opt.scaling) {
            sc_ = compute_scaling(p.A, opt.ruiz_iterations, opt.pock_chambolle_alpha);
            q_ = apply_scaling(p, sc_);
        } else {
            sc_.row.assign(m_, 1.0);
            sc_.col.assign(n_, 1.0);
            q_ = p;
        }
        A_ = &q_.A;
        AT_ = q_.A.transpose();
        lb_.assign(N_, 0.0);
        ub_.assign(N_, 0.0);
        cost0_.assign(N_, 0.0);
        for (int j = 0; j < n_; ++j) cost0_[j] = q_.c[j];
        for (int i = 0; i < m_; ++i) {
            lb_[n_ + i] = q_.row_lb[i];
            ub_[n_ + i] = q_.row_ub[i];
        }
        slack_idx_.resize(m_);
        for (int i = 0; i < m_; ++i) slack_idx_[i] = i;
        slack_val_.assign(m_, -1.0);
        prepared_ = true;
    }

    // Solve with the given column bounds (original scale) instead of the problem's,
    // from `warm` if given; `out` receives the final basis. light: skip the residual
    // evaluation on the original problem (x, y and the objective are still set).
    LpSolution
    resolve(const std::vector<double> &col_lb, const std::vector<double> &col_ub,
        const SimplexBasis *warm = nullptr, SimplexBasis *out = nullptr,
        const LpSolution *start = nullptr, bool light = true)
    {
        if (!prepared_) throw std::logic_error("DualSimplex::resolve before prepare");
        t_start_ = std::chrono::steady_clock::now();
        for (int j = 0; j < n_; ++j) {
            lb_[j] = col_lb[j] / sc_.col[j];
            ub_[j] = col_ub[j] / sc_.col[j];
        }
        cost_ = cost0_;
        x_.assign(N_, 0.0);
        d_.assign(N_, 0.0);
        y_.assign(m_, 0.0);
        status_.assign(N_, VarStatus::AtLower);
        head_.assign(m_, -1);
        pos_.assign(N_, -1);
        dse_.assign(m_, 1.0);
        iters_ = 0;
        perturbed_ = false;
        rng_.seed(12345);

        // crossover: derive a starting basis from an interior point
        SimplexBasis from_point;
        crossover_ = false;
        if (start && !warm && start->x.size() == static_cast<size_t>(n_) &&
            start->y.size() == static_cast<size_t>(m_)) {
            from_point = basis_from_point(*start);
            warm = &from_point;
            crossover_ = true;
        }
        Status st = run(warm);
        LpSolution sol = extract(st, light);
        if (out) out->status = status_;
        return sol;
    }

    long iterations() const { return iters_; }

  private:
    // ---- problem data (scaled) --------------------------------------------
    const LpProblem *orig_ = nullptr;
    const SolverOptions *opt_ = nullptr;
    LpProblem q_;
    Scaling sc_;
    const HostMatrix *A_ = nullptr;
    HostMatrix AT_;
    int n_ = 0, m_ = 0, N_ = 0;
    std::vector<double> lb_, ub_, cost0_, cost_;
    std::vector<int> slack_idx_;
    std::vector<double> slack_val_;
    // ---- simplex state ------------------------------------------------------
    std::vector<double> x_, d_, y_, dse_, pinf_;
    std::vector<uint8_t> active_;
    std::vector<std::vector<int>> price_buf_;
    std::vector<int> cbuf_;
    std::vector<VarStatus> status_;
    std::vector<int> head_, pos_;
    BasisFactor bf_;
    long iters_ = 0;
    bool perturbed_ = false, dse_bad_ = false, crossover_ = false, prepared_ = false;
    long dse_rebuild_iter_ = -1000;
    std::mt19937_64 rng_;
    std::chrono::steady_clock::time_point t_start_;
    const double ptol_ = 1e-7, dtol_ = 1e-7;
    // DSE weights must stay finite and positive or CHUZR silently skips rows
    static double clamp_dse(double w) { return std::isfinite(w) ? std::min(std::max(w, 1e-4), 1e12) : 1e12; }

    double
    elapsed() const
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
    }

    bool isfin(double v) const { return std::abs(v) < 1e300; }

    BasisFactor::Column
    column(int j) const
    {
        if (j < n_)
            return {AT_.col_ind() + AT_.row_ptr()[j], AT_.values() + AT_.row_ptr()[j],
                AT_.row_ptr()[j + 1] - AT_.row_ptr()[j]};
        return {&slack_idx_[j - n_], &slack_val_[j - n_], 1};
    }

    // Value of a nonbasic variable given its status and the bound arrays.
    double
    nb_value(int j, const std::vector<double> &LB, const std::vector<double> &UB) const
    {
        switch (status_[j]) {
        case VarStatus::AtLower: return isfin(LB[j]) ? LB[j] : 0.0;
        case VarStatus::AtUpper: return isfin(UB[j]) ? UB[j] : 0.0;
        default: return 0.0;
        }
    }

    // Crossover start: rank the variables by how far they are from their bounds
    // relative to their reduced cost (interior in the primal, not in the dual =>
    // likely basic) and take the m best as the initial basis; the rest go to
    // their nearest bound. Dependent columns are repaired by refactor().
    SimplexBasis
    basis_from_point(const LpSolution &s) const
    {
        std::vector<double> xs(N_, 0.0), ys(m_), ds(N_, 0.0);
        for (int j = 0; j < n_; ++j) xs[j] = s.x[j] / sc_.col[j];
        for (int i = 0; i < m_; ++i) ys[i] = s.y[i] / sc_.row[i];
        for (int i = 0; i < m_; ++i)
            for (int k = A_->row_ptr()[i]; k < A_->row_ptr()[i + 1]; ++k)
                xs[n_ + i] += A_->values()[k] * xs[A_->col_ind()[k]];
        for (int j = 0; j < n_; ++j) {
            double v = cost0_[j];
            for (int k = AT_.row_ptr()[j]; k < AT_.row_ptr()[j + 1]; ++k)
                v -= AT_.values()[k] * ys[AT_.col_ind()[k]];
            ds[j] = v;
        }
        for (int i = 0; i < m_; ++i) ds[n_ + i] = ys[i];
        std::vector<double> score(N_);
        for (int j = 0; j < N_; ++j) {
            if (lb_[j] == ub_[j]) { score[j] = -1.0; continue; }
            double dp = 1e30;
            if (isfin(lb_[j])) dp = std::min(dp, xs[j] - lb_[j]);
            if (isfin(ub_[j])) dp = std::min(dp, ub_[j] - xs[j]);
            dp = std::max(dp, 0.0);
            score[j] = dp >= 1e30 ? 2.0 : dp / (dp + std::abs(ds[j]) + 1e-12);
        }
        std::vector<int> order(N_);
        for (int j = 0; j < N_; ++j) order[j] = j;
        std::sort(order.begin(), order.end(),
            [&](int a, int b) { return score[a] > score[b]; });
        // Take the m best-ranked columns; columns that turn out dependent are
        // replaced by the next-best ones (slacks of the unpivoted rows at the end).
        std::vector<int> head(order.begin(), order.begin() + m_);
        std::vector<char> in(N_, 0);
        for (int j : head) in[j] = 1;
        size_t next = m_;
        {
            BasisFactor bf;
            std::vector<int> sing, freerow;
            for (int round = 0; round < 40; ++round) {
                std::vector<BasisFactor::Column> cols(m_);
                for (int k = 0; k < m_; ++k) cols[k] = column(head[k]);
                const int ns = bf.factor(m_, cols, sing, freerow);
                if (ns == 0) break;
                for (int t = 0; t < ns; ++t) {
                    in[head[sing[t]]] = 0;
                    int repl = -1;
                    while (next < order.size() && in[order[next]]) ++next;
                    if (next < order.size() && round < 39) repl = order[next++];
                    else if (!in[n_ + freerow[t]]) repl = n_ + freerow[t];
                    if (repl < 0) repl = head[sing[t]]; // give up; refactor() repairs
                    head[sing[t]] = repl;
                    in[repl] = 1;
                }
            }
        }
        SimplexBasis b;
        b.status.assign(N_, VarStatus::AtLower);
        for (int j = 0; j < N_; ++j) {
            const bool fl = isfin(lb_[j]), fu = isfin(ub_[j]);
            if (!fl && !fu) b.status[j] = VarStatus::AtZero;
            else if (fl && fu) b.status[j] = (xs[j] - lb_[j] <= ub_[j] - xs[j]) ? VarStatus::AtLower : VarStatus::AtUpper;
            else b.status[j] = fl ? VarStatus::AtLower : VarStatus::AtUpper;
        }
        for (int j : head) b.status[j] = VarStatus::Basic;
        if (opt_->verbose) {
            int strong = 0, weak = 0;
            for (int j = 0; j < N_; ++j) { strong += score[j] > 0.9; weak += score[j] > 0.5; }
            std::printf("[simplex] crossover: m %d, score>0.9: %d, score>0.5: %d\n", m_, strong, weak);
        }
        return b;
    }

    // ---- factorization ---------------------------------------------------
    // Factor the current basis, repairing singular columns with slacks.
    bool
    refactor()
    {
        for (int attempt = 0; attempt < 5; ++attempt) {
            std::vector<BasisFactor::Column> cols(m_);
            for (int k = 0; k < m_; ++k) cols[k] = column(head_[k]);
            std::vector<int> sing, freerow;
            int ns = bf_.factor(m_, cols, sing, freerow);
            if (ns == 0) return true;
            for (int t = 0; t < ns; ++t) {
                const int k = sing[t];
                const int leaving = head_[k];
                const int entering = n_ + freerow[t];
                // the slack of a free row may already be basic elsewhere
                if (pos_[entering] >= 0) continue;
                head_[k] = entering;
                pos_[entering] = k;
                status_[entering] = VarStatus::Basic;
                pos_[leaving] = -1;
                status_[leaving] = isfin(lb_[leaving]) ? VarStatus::AtLower
                                   : isfin(ub_[leaving]) ? VarStatus::AtUpper
                                                         : VarStatus::AtZero;
            }
        }
        return false;
    }

    // x_B from the nonbasic values (bounds arrays chosen by the caller).
    void
    compute_primal(const std::vector<double> &LB, const std::vector<double> &UB)
    {
        std::vector<double> rhs(m_, 0.0);
        for (int j = 0; j < N_; ++j) {
            if (status_[j] == VarStatus::Basic) continue;
            const double v = nb_value(j, LB, UB);
            x_[j] = v;
            if (v == 0.0) continue;
            BasisFactor::Column c = column(j);
            for (int t = 0; t < c.nnz; ++t) rhs[c.idx[t]] -= c.val[t] * v;
        }
        bf_.ftran(rhs);
        for (int k = 0; k < m_; ++k) x_[head_[k]] = rhs[k];
    }

    void
    compute_dual()
    {
        std::vector<double> cb(m_);
        for (int k = 0; k < m_; ++k) cb[k] = cost_[head_[k]];
        bf_.btran(cb);
        y_ = cb;
        for (int j = 0; j < N_; ++j) {
            if (status_[j] == VarStatus::Basic) { d_[j] = 0.0; continue; }
            BasisFactor::Column c = column(j);
            double s = cost_[j];
            for (int t = 0; t < c.nnz; ++t) s -= c.val[t] * y_[c.idx[t]];
            d_[j] = s;
        }
    }

    // ---- dual feasibility ------------------------------------------------
    // Choose nonbasic bounds that make the basis dual feasible where the
    // bounds allow it. Returns the number of variables that stay infeasible.
    int
    place_nonbasics(const std::vector<double> &LB, const std::vector<double> &UB)
    {
        int infeasible = 0;
        for (int j = 0; j < N_; ++j) {
            if (status_[j] == VarStatus::Basic) continue;
            const double l = LB[j], u = UB[j];
            if (l == u) { status_[j] = VarStatus::AtLower; continue; }
            const double dj = d_[j];
            if (dj > dtol_) {
                if (isfin(l)) status_[j] = VarStatus::AtLower;
                else ++infeasible;
            } else if (dj < -dtol_) {
                if (isfin(u)) status_[j] = VarStatus::AtUpper;
                else ++infeasible;
            } else { // (nearly) zero reduced cost: keep a valid bound
                if (status_[j] == VarStatus::AtLower && isfin(l)) {}
                else if (status_[j] == VarStatus::AtUpper && isfin(u)) {}
                else status_[j] = isfin(l) ? VarStatus::AtLower
                                  : isfin(u) ? VarStatus::AtUpper
                                             : VarStatus::AtZero;
            }
        }
        return infeasible;
    }

    // Exact dual steepest-edge weights ||e_k^T B^-1||^2 for the current basis.
    void
    rebuild_dse()
    {
        dse_rebuild_iter_ = iters_;
        SVec e;
        e.init(m_);
        for (int k = 0; k < m_; ++k) {
            // m BTRANs can take long on big bases: keep the old weights for the
            // rest when out of time or interrupted
            if ((k & 255) == 0 && (elapsed() > opt_->time_limit || opt_->interrupted())) return;
            e.clear();
            e.set(k, 1.0);
            bf_.btran(e);
            double n2 = 0;
            for (int i : e.idx) n2 += e.v[i] * e.v[i];
            dse_[k] = clamp_dse(n2);
        }
    }

    // active_[j]: nonbasic and not fixed for the bounds of the running loop.
    // Only these entries of a pivot row are ever used.
    void
    rebuild_active(const std::vector<double> &LB, const std::vector<double> &UB)
    {
        active_.resize(N_);
        for (int j = 0; j < N_; ++j)
            active_[j] = status_[j] != VarStatus::Basic && LB[j] != UB[j];
    }

    // rho = B^-T e_r and the pivot row arow = rho^T [A -I], built sparsely:
    // aidx lists the (possibly) nonzero entries of arow, active columns only.
    void
    pivot_row(int r, SVec &rho, std::vector<double> &arow, std::vector<int> &aidx)
    {
        rho.clear();
        rho.set(r, 1.0);
        bf_.btran(rho);
        for (int j : aidx) arow[j] = 0.0;
        aidx.clear();
        // row-wise product costs the row lengths of rho's nonzeros, column-wise nnz(A)
        size_t rowwork = 0;
        for (int i : rho.idx) rowwork += A_->row_ptr()[i + 1] - A_->row_ptr()[i];
        if (rowwork * 3 < static_cast<size_t>(A_->nnz())) { // loop over the rows rho touches
            const int *rp = A_->row_ptr(), *ci = A_->col_ind();
            const double *va = A_->values();
            for (int i : rho.idx) {
                const double ri = rho.v[i];
                if (ri == 0.0) continue;
                for (int k = rp[i]; k < rp[i + 1]; ++k) {
                    const int j = ci[k];
                    if (!active_[j]) continue; // basic or fixed: never needed
                    if (arow[j] == 0.0) aidx.push_back(j);
                    arow[j] += ri * va[k];
                    if (arow[j] == 0.0) arow[j] = 1e-300; // keep the pattern entry
                }
            }
        } else {
            const int *rp = AT_.row_ptr(), *ci = AT_.col_ind();
            const double *va = AT_.values();
            // one pass over all of A: split over threads for large problems
            auto cols = [&](int j0, int j1, std::vector<int> &out) {
                for (int j = j0; j < j1; ++j) {
                    if (!active_[j]) continue;
                    double sj = 0;
                    for (int k = rp[j]; k < rp[j + 1]; ++k) sj += rho.v[ci[k]] * va[k];
                    if (sj != 0.0) { arow[j] = sj; out.push_back(j); }
                }
            };
#ifdef _OPENMP
            const int T = omp_get_max_threads();
            if (T > 1 && A_->nnz() > 200000) {
                price_buf_.resize(T);
#pragma omp parallel num_threads(T)
                {
                    const int t = omp_get_thread_num(), nt = omp_get_num_threads();
                    std::vector<int> &out = price_buf_[t];
                    out.clear();
                    const int j0 = static_cast<int>(static_cast<long>(n_) * t / nt);
                    const int j1 = static_cast<int>(static_cast<long>(n_) * (t + 1) / nt);
                    cols(j0, j1, out);
                }
                for (auto &b : price_buf_) aidx.insert(aidx.end(), b.begin(), b.end());
            } else
#endif
                cols(0, n_, aidx);
        }
        for (int i : rho.idx)
            if (rho.v[i] != 0.0 && active_[n_ + i]) { arow[n_ + i] = -rho.v[i]; aidx.push_back(n_ + i); }
    }

    // squared primal infeasibility of the basic variable at position k
    void
    set_pinf(int k, const std::vector<double> &LB, const std::vector<double> &UB)
    {
        const int pv = head_[k];
        const double xv = x_[pv];
        if (xv < LB[pv] - ptol_) pinf_[k] = (LB[pv] - xv) * (LB[pv] - xv);
        else if (xv > UB[pv] + ptol_) pinf_[k] = (xv - UB[pv]) * (xv - UB[pv]);
        else pinf_[k] = 0.0;
    }

    void
    recompute_pinf(const std::vector<double> &LB, const std::vector<double> &UB)
    {
        pinf_.assign(m_, 0.0);
        for (int k = 0; k < m_; ++k) set_pinf(k, LB, UB);
    }

    // ---- the dual simplex loop -------------------------------------------
    // Returns Optimal (primal feasible for LB/UB), Infeasible (dual
    // unbounded), or a limit / numerical status.
    Status
    dual_loop(const std::vector<double> &LB, const std::vector<double> &UB, bool phase1)
    {
        const int refactor_max = 2000; // safety cap on eta-file length
        RefactorPolicy policy;
        policy.margin = m_ > 10000 ? 2.0 : 1.0;
        policy.fixed_freq = opt_->deterministic ? 100 : 0;
        bool policy_refactor = false;
        SVec rho, colq, tau, rhs;
        rho.init(m_); colq.init(m_); tau.init(m_); rhs.init(m_);
        std::vector<double> arow(N_, 0.0);
        std::vector<int> cand, flips, aidx;
        std::vector<std::pair<double, int>> heap;
        long degenerate = 0;
        bool need_refactor = true;
        bool fresh = false;
        int bad_iters = 0;
        std::vector<int> chuzr_list;
        std::vector<char> in_list(m_, 0);
        double chuzr_cutoff = 0.0;
        bool need_full_scan = true;
        const size_t kList = 64;
        auto full_scan = [&]() {
            std::vector<std::pair<double, int>> sc;
            for (int k = 0; k < m_; ++k) {
                if (pinf_[k] == 0.0) continue;
                sc.emplace_back(pinf_[k] / dse_[k], k);
            }
            for (int k : chuzr_list) in_list[k] = 0;
            chuzr_list.clear();
            chuzr_cutoff = 0.0;
            if (sc.size() > kList) {
                std::nth_element(sc.begin(), sc.begin() + kList, sc.end(),
                    [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                        return a.first > b.first;
                    });
                chuzr_cutoff = sc[kList].first; // everything outside the list scores <= this
                sc.resize(kList);
            }
            for (auto &e : sc) { chuzr_list.push_back(e.second); in_list[e.second] = 1; }
        };

        while (true) {
            if (iters_ >= opt_->max_iterations) return Status::IterationLimit;
            if ((iters_ & 15) == 0 && elapsed() > opt_->time_limit) return Status::TimeLimit;
            if ((iters_ & 15) == 0 && opt_->interrupted()) return Status::Interrupted;
            if (need_refactor || policy_refactor || bf_.eta_heavy(8.0)) {
                const auto tf0 = std::chrono::steady_clock::now();
                if (!refactor()) return Status::NumericalError;
                policy_refactor = false;
                compute_primal(LB, UB);
                compute_dual();
                rebuild_active(LB, UB);
                policy.begin_cycle(std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count());
                need_refactor = false;
                fresh = true;
                if (dse_bad_) { dse_bad_ = false; rebuild_dse(); }
                // small dual infeasibilities from drift: shift costs
                for (int j = 0; j < N_; ++j) {
                    if (status_[j] == VarStatus::Basic) continue;
                    if (LB[j] == UB[j]) continue;
                    double shift = 0;
                    if (status_[j] == VarStatus::AtLower && d_[j] < 0) shift = -d_[j];
                    else if (status_[j] == VarStatus::AtUpper && d_[j] > 0) shift = -d_[j];
                    else if (status_[j] == VarStatus::AtZero && d_[j] != 0) shift = -d_[j];
                    if (shift != 0) { cost_[j] += shift; d_[j] += shift; }
                }
                recompute_pinf(LB, UB);
                need_full_scan = true;
            }
            // ---- CHUZR: leaving row -------------------------------------
            // Candidate list of the largest scores pinf/dse: only positions touched
            // by the last update can beat the cutoff of the last full scan, so the
            // O(m) scan is repeated only when the list runs dry.
            auto score_of = [&](int k) { return pinf_[k] == 0.0 ? 0.0 : pinf_[k] / dse_[k]; };
            int r = -1;
            double best = 0;
            for (int attempt = 0; attempt < 2 && r < 0; ++attempt) {
                if (need_full_scan) {
                    full_scan();
                    need_full_scan = false;
                }
                size_t w = 0;
                for (size_t t = 0; t < chuzr_list.size(); ++t) {
                    const int k = chuzr_list[t];
                    const double sc = score_of(k);
                    if (sc == 0.0) { in_list[k] = 0; continue; }
                    chuzr_list[w++] = k;
                    if (sc > best) { best = sc; r = k; }
                }
                chuzr_list.resize(w);
                if (r >= 0 && (best >= chuzr_cutoff || chuzr_cutoff == 0.0)) break;
                if (attempt == 0 && chuzr_cutoff == 0.0) break; // the list was complete
                r = -1;
                best = 0;
                need_full_scan = true;
            }
            if (r < 0) {
                if (!fresh) { need_refactor = true; continue; } // confirm on a clean factor
                return Status::Optimal;
            }
            const int p = head_[r];
            const bool below = x_[p] < LB[p];
            const double delta = below ? x_[p] - LB[p] : x_[p] - UB[p];

            // ---- BTRAN and pivot row ----------------------------------------
            pivot_row(r, rho, arow, aidx);
            // ---- candidates for entering ------------------------------------
            // branch-free scan (the status/sign test is unpredictable)
            if (cbuf_.size() < aidx.size()) cbuf_.resize(N_); // grown once, never zero-filled again
            double amax = 0;
            {
                const double sgn = below ? -1.0 : 1.0;
                size_t w = 0;
                int *cb = cbuf_.data();
                for (int j : aidx) { // active entries only: nonbasic, not fixed
                    const double a = arow[j];
                    amax = std::max(amax, std::abs(a));
                    const double at = sgn * a;
                    const VarStatus st = status_[j];
                    const bool keep = (st == VarStatus::AtLower) & (at > 0);
                    const bool keep2 = (st == VarStatus::AtUpper) & (at < 0);
                    const bool keep3 = (st == VarStatus::AtZero) & (at != 0);
                    cb[w] = j;
                    w += static_cast<size_t>(keep | keep2 | keep3);
                }
                cand.assign(cb, cb + w);
            }
            {
                const double pivtol = 1e-7 * std::max(1.0, amax);
                size_t w = 0;
                for (int j : cand)
                    if (std::abs(arow[j]) > pivtol) cand[w++] = j;
                cand.resize(w);
            }
            if (cand.empty()) return Status::Infeasible;
            auto ratio = [&](int j) {
                const double at = below ? -arow[j] : arow[j];
                return std::max(0.0, d_[j] / at); // >= 0 for a dual feasible j
            };
            // lazy ordering: min-heap on the ratio, pop only as far as needed
            heap.clear();
            for (int j : cand) heap.emplace_back(ratio(j), j);
            const auto hcmp = [](const std::pair<double, int> &x,
                                  const std::pair<double, int> &y) { return x.first > y.first; };
            std::make_heap(heap.begin(), heap.end(), hcmp);
            size_t hend = heap.size(); // heap occupies [0, hend); popped sit behind it
            // ---- bound-flipping ratio test -------------------------------
            double slope = std::abs(delta);
            flips.clear();
            bool found = false;
            while (hend > 0) {
                const int j = heap.front().second;
                const double range = UB[j] - LB[j];
                if (!isfin(range)) { found = true; break; } // non-boxed: must enter
                const double at = std::abs(arow[j]);
                const double ns = slope - at * range;
                if (ns <= ptol_) { found = true; break; } // slope used up
                slope = ns;
                flips.push_back(j);
                std::pop_heap(heap.begin(), heap.begin() + hend, hcmp);
                --hend;
            }
            if (!found) return Status::Infeasible; // dual unbounded
            // Harris choice among the remaining candidates heap[0, hend)
            // The tolerance grows while the best pivot is small relative to the row:
            // wrong-signed reduced costs this creates are shifted away below.
            double amax_rem = 0;
            for (size_t t = 0; t < hend; ++t) amax_rem = std::max(amax_rem, std::abs(arow[heap[t].second]));
            int q = heap.front().second;
            for (double tol = dtol_;; tol *= 10) {
                double theta_max = 1e300;
                for (size_t t = 0; t < hend; ++t) {
                    const int j = heap[t].second;
                    theta_max = std::min(theta_max, (std::abs(d_[j]) + tol) / std::abs(arow[j]));
                }
                double qabs = 0;
                for (size_t t = 0; t < hend; ++t) {
                    const int j = heap[t].second;
                    const double at = std::abs(arow[j]);
                    if (heap[t].first > theta_max) continue;
                    if (at > qabs) { qabs = at; q = j; }
                }
                if (qabs >= 1e-2 * amax_rem || tol >= 1e-5) break;
            }
            const double alpha_q = arow[q];
            const double theta_d = d_[q] / alpha_q;

            // ---- FTRAN of the entering column ---------------------------------
            colq.clear();
            {
                BasisFactor::Column c = column(q);
                for (int t = 0; t < c.nnz; ++t) colq.set(c.idx[t], c.val[t]);
            }
            bf_.ftran(colq);
            if (std::abs(colq.v[r] - alpha_q) > 1e-6 * (1.0 + std::abs(alpha_q)) ||
                std::abs(colq.v[r]) < 1e-11) {
                // the pivot row and column disagree: numerical trouble
                if (bf_.updates() == 0 && ++bad_iters > 5) return Status::NumericalError;
                need_refactor = true;
                continue;
            }
            fresh = false;
            bad_iters = 0;

            // ---- bound flips: update x_B for the moved nonbasics -----------
            if (!flips.empty()) {
                rhs.clear();
                for (int j : flips) {
                    const bool to_upper = status_[j] == VarStatus::AtLower;
                    const double nv = to_upper ? UB[j] : LB[j];
                    const double dx = nv - x_[j];
                    x_[j] = nv;
                    status_[j] = to_upper ? VarStatus::AtUpper : VarStatus::AtLower;
                    BasisFactor::Column c = column(j);
                    for (int t = 0; t < c.nnz; ++t) rhs.add(c.idx[t], c.val[t] * dx);
                }
                bf_.ftran(rhs);
                for (int k : rhs.idx) {
                    x_[head_[k]] -= rhs.v[k];
                    set_pinf(k, LB, UB);
                }
            }
            // ---- primal step ------------------------------------------------
            const double target = below ? LB[p] : UB[p];
            const double theta_p = (x_[p] - target) / colq.v[r];
            for (int k : colq.idx) {
                if (colq.v[k] == 0.0) continue;
                x_[head_[k]] -= theta_p * colq.v[k];
                set_pinf(k, LB, UB);
            }
            x_[q] += theta_p;
            x_[p] = target;

            // ---- dual update ----------------------------------------------------
            for (int j : aidx) d_[j] -= theta_d * arow[j]; // active entries only
            d_[q] = 0.0;
            d_[p] = -theta_d;
            // wrong-signed reduced costs left by the Harris step: shift costs
            for (size_t t = 0; t < hend; ++t) {
                const int j = heap[t].second;
                if (j == q) continue;
                const VarStatus st = status_[j];
                if ((st == VarStatus::AtLower && d_[j] < 0) ||
                    (st == VarStatus::AtUpper && d_[j] > 0)) {
                    cost_[j] -= d_[j];
                    d_[j] = 0.0;
                }
            }
            degenerate = std::abs(theta_d) < 1e-12 ? degenerate + 1 : 0;

            // ---- DSE weights ------------------------------------------------------
            tau.clear();
            double wr = 0;
            for (int i : rho.idx) {
                tau.set(i, rho.v[i]);
                wr += rho.v[i] * rho.v[i];
            }
            bf_.ftran(tau);
            // the exact weight of the leaving row is known from rho: use it (this
            // also repairs the drifting updated weights, as in HiGHS)
            wr = clamp_dse(wr);
            // the updated weight was far off: the weights have lost accuracy, rebuild
            if (dse_[r] > 1000 * wr || dse_[r] * 1000 < wr) {
                if (iters_ - dse_rebuild_iter_ > 30) dse_bad_ = true;
            }
            const double ar = colq.v[r];
            for (int k : colq.idx) {
                if (k == r || colq.v[k] == 0.0) continue;
                const double ratio_k = colq.v[k] / ar;
                dse_[k] = clamp_dse(dse_[k] - 2.0 * ratio_k * tau.v[k] + ratio_k * ratio_k * wr);
            }
            dse_[r] = clamp_dse(wr / (ar * ar));

            // ---- basis change --------------------------------------------------------
            head_[r] = q;
            pos_[q] = r;
            pos_[p] = -1;
            active_[q] = 0;
            active_[p] = LB[p] != UB[p];
            status_[q] = VarStatus::Basic;
            status_[p] = (LB[p] == UB[p] || below) ? VarStatus::AtLower : VarStatus::AtUpper;
            set_pinf(r, LB, UB);
            {   // positions whose score changed may now beat the cutoff
                auto touch = [&](int k) {
                    if (in_list[k]) return;
                    const double sc = score_of(k);
                    if (sc > chuzr_cutoff) { chuzr_list.push_back(k); in_list[k] = 1; }
                };
                for (int k : colq.idx) touch(k);
                if (!flips.empty())
                    for (int k : rhs.idx) touch(k);
                touch(r);
                if (chuzr_list.size() > 4 * kList) need_full_scan = true;
            }
            if (!bf_.update(r, colq)) need_refactor = true;
            ++iters_;
            if (dse_bad_) need_refactor = true;
            if (policy.tick(refactor_max)) policy_refactor = true;
            if (opt_->verbose && (iters_ % 200 == 0))
                std::printf("[simplex] it %ld  |infeas| %.3e  theta_d %.2e  etas %d%s\n",
                    iters_, std::abs(delta), theta_d, bf_.updates(),
                    phase1 ? "  (phase 1)" : "");

            // ---- stall handling: perturb costs ------------------------------------
            if (!perturbed_ && degenerate > 60) perturb(); // also in phase 1: it has the true costs
        }
    }

    // ---- the primal simplex loop ---------------------------------------------
    // Starts from a primal feasible basis (x_, d_, status_ consistent) and removes
    // dual infeasibilities: used after the dual simplex ran on shifted costs, and
    // as a cleanup. Devex pricing, Harris ratio test with bound flipping of the
    // entering variable.
    Status
    primal_loop(const std::vector<double> &LB, const std::vector<double> &UB)
    {
        const int refactor_max = 2000;
        RefactorPolicy policy;
        policy.margin = m_ > 10000 ? 2.0 : 1.0;
        policy.fixed_freq = opt_->deterministic ? 100 : 0;
        bool policy_refactor = false;
        SVec rho;
        rho.init(m_);
        std::vector<double> colq(m_), arow(N_, 0.0), w(N_, 1.0);
        std::vector<int> aidx;
        bool need_refactor = true, fresh = false;
        int bad_iters = 0;
        while (true) {
            if (iters_ >= opt_->max_iterations) return Status::IterationLimit;
            if ((iters_ & 63) == 0 && elapsed() > opt_->time_limit) return Status::TimeLimit;
            if ((iters_ & 15) == 0 && opt_->interrupted()) return Status::Interrupted;
            if (need_refactor || policy_refactor || bf_.eta_heavy(8.0)) {
                const auto tf0 = std::chrono::steady_clock::now();
                if (!refactor()) return Status::NumericalError;
                policy_refactor = false;
                compute_primal(LB, UB);
                compute_dual();
                rebuild_active(LB, UB);
                policy.begin_cycle(std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count());
                need_refactor = false;
                fresh = true;
            }
            // ---- CHUZC: entering variable ---------------------------------
            int q = -1;
            double best = 0;
            for (int j = 0; j < N_; ++j) {
                const VarStatus st = status_[j];
                if (st == VarStatus::Basic || LB[j] == UB[j]) continue;
                const double dj = d_[j];
                double viol = 0;
                if (st == VarStatus::AtLower) { if (dj < -dtol_) viol = -dj; }
                else if (st == VarStatus::AtUpper) { if (dj > dtol_) viol = dj; }
                else if (std::abs(dj) > dtol_) viol = std::abs(dj);
                if (viol == 0) continue;
                const double score = viol * viol / w[j];
                if (score > best) { best = score; q = j; }
            }
            if (q < 0) {
                if (!fresh) { need_refactor = true; continue; }
                return Status::Optimal;
            }
            const double dir = (status_[q] == VarStatus::AtUpper ||
                                   (status_[q] == VarStatus::AtZero && d_[q] > 0)) ? -1.0 : 1.0;
            std::fill(colq.begin(), colq.end(), 0.0);
            {
                BasisFactor::Column c = column(q);
                for (int t = 0; t < c.nnz; ++t) colq[c.idx[t]] = c.val[t];
            }
            bf_.ftran(colq);
            double amax = 0;
            for (int k = 0; k < m_; ++k) amax = std::max(amax, std::abs(colq[k]));
            const double pivtol = 1e-9 * std::max(1.0, amax);
            // ---- Harris ratio test ------------------------------------------
            // x_B(k) changes at the rate -dir * colq[k] per unit step
            double tmax = 1e300;
            for (int k = 0; k < m_; ++k) {
                const double a = dir * colq[k];
                if (std::abs(a) <= pivtol) continue;
                const int pv = head_[k];
                if (a > 0) { if (isfin(LB[pv])) tmax = std::min(tmax, (x_[pv] - LB[pv] + ptol_) / a); }
                else if (isfin(UB[pv])) tmax = std::min(tmax, (UB[pv] - x_[pv] + ptol_) / -a);
            }
            int r = -1;
            double bestabs = 0, thetap = 0;
            if (tmax < 1e300) {
                for (int k = 0; k < m_; ++k) {
                    const double a = dir * colq[k];
                    if (std::abs(a) <= pivtol) continue;
                    const int pv = head_[k];
                    double t;
                    if (a > 0) { if (!isfin(LB[pv])) continue; t = (x_[pv] - LB[pv]) / a; }
                    else { if (!isfin(UB[pv])) continue; t = (UB[pv] - x_[pv]) / -a; }
                    if (t > tmax) continue;
                    if (std::abs(a) > bestabs) { bestabs = std::abs(a); r = k; thetap = std::max(t, 0.0); }
                }
            }
            const double rng = UB[q] - LB[q];
            if (r < 0 && !isfin(rng)) return Status::Unbounded;
            if (r < 0 || (isfin(rng) && rng <= thetap)) { // bound flip of the entering variable
                const double th = rng;
                for (int k = 0; k < m_; ++k)
                    if (colq[k] != 0.0) x_[head_[k]] -= dir * th * colq[k];
                const bool to_upper = dir > 0;
                x_[q] = to_upper ? UB[q] : LB[q];
                status_[q] = to_upper ? VarStatus::AtUpper : VarStatus::AtLower;
                fresh = false;
                ++iters_;
                continue;
            }
            // ---- pivot row for the dual update -----------------------------
            pivot_row(r, rho, arow, aidx);
            const double alpha_r = arow[q];
            if (std::abs(alpha_r - colq[r]) > 1e-6 * (1.0 + std::abs(colq[r])) ||
                std::abs(colq[r]) < 1e-11) {
                if (bf_.updates() == 0 && ++bad_iters > 5) return Status::NumericalError;
                need_refactor = true;
                continue;
            }
            fresh = false;
            bad_iters = 0;
            const int p = head_[r];
            const bool p_to_lower = dir * colq[r] > 0;
            for (int k = 0; k < m_; ++k)
                if (colq[k] != 0.0) x_[head_[k]] -= dir * thetap * colq[k];
            x_[q] += dir * thetap;
            x_[p] = p_to_lower ? LB[p] : UB[p];
            const double theta_d = d_[q] / alpha_r;
            for (int j : aidx) {
                if (status_[j] == VarStatus::Basic) continue;
                d_[j] -= theta_d * arow[j];
            }
            d_[q] = 0.0;
            d_[p] = -theta_d;
            // devex reference weights
            const double wq = w[q];
            for (int j : aidx) {
                if (status_[j] == VarStatus::Basic || j == q) continue;
                const double ratio = arow[j] / alpha_r;
                w[j] = std::max(w[j], ratio * ratio * wq);
            }
            w[p] = std::max(wq / (alpha_r * alpha_r), 1.0);
            if (w[p] > 1e6) std::fill(w.begin(), w.end(), 1.0);
            head_[r] = q;
            pos_[q] = r;
            pos_[p] = -1;
            active_[q] = 0;
            active_[p] = LB[p] != UB[p];
            status_[q] = VarStatus::Basic;
            status_[p] = (LB[p] == UB[p] || p_to_lower) ? VarStatus::AtLower : VarStatus::AtUpper;
            if (!bf_.update(r, colq)) need_refactor = true;
            ++iters_;
            if (policy.tick(refactor_max)) policy_refactor = true;
            if (opt_->verbose && (iters_ % 200 == 0))
                std::printf("[simplex] it %ld  primal  d_q %.3e  etas %d\n", iters_, d_[q], bf_.updates());
        }
    }

    // Small random cost perturbation in the dual-feasible direction.
    void
    perturb()
    {
        perturbed_ = true;
        std::uniform_real_distribution<double> u(0.5, 1.0);
        for (int j = 0; j < N_; ++j) {
            const VarStatus st = status_[j];
            if (st == VarStatus::Basic || lb_[j] == ub_[j]) continue;
            const double e = u(rng_) * 5e-7 * (1.0 + std::abs(cost0_[j]));
            if (st == VarStatus::AtLower) { cost_[j] += e; d_[j] += e; }
            else if (st == VarStatus::AtUpper) { cost_[j] -= e; d_[j] -= e; }
        }
    }

    // ---- driver ------------------------------------------------------------
    Status
    run(const SimplexBasis *warm)
    {
        // initial basis
        bool have_basis = false;
        if (warm && warm->valid() && static_cast<int>(warm->status.size()) == N_) {
            int nb = 0;
            for (VarStatus s : warm->status) nb += (s == VarStatus::Basic);
            if (nb == m_) {
                status_ = warm->status;
                int k = 0;
                for (int j = 0; j < N_; ++j)
                    if (status_[j] == VarStatus::Basic) { head_[k] = j; pos_[j] = k; ++k; }
                for (int j = 0; j < N_; ++j) {
                    if (status_[j] == VarStatus::Basic) continue;
                    // repair statuses that need an infinite bound
                    if (status_[j] == VarStatus::AtLower && !isfin(lb_[j]))
                        status_[j] = isfin(ub_[j]) ? VarStatus::AtUpper : VarStatus::AtZero;
                    if (status_[j] == VarStatus::AtUpper && !isfin(ub_[j]))
                        status_[j] = isfin(lb_[j]) ? VarStatus::AtLower : VarStatus::AtZero;
                }
                have_basis = true;
            }
        }
        if (!have_basis) {
            for (int j = 0; j < n_; ++j) {
                status_[j] = isfin(lb_[j]) ? VarStatus::AtLower
                             : isfin(ub_[j]) ? VarStatus::AtUpper : VarStatus::AtZero;
                pos_[j] = -1;
            }
            for (int i = 0; i < m_; ++i) {
                head_[i] = n_ + i;
                pos_[n_ + i] = i;
                status_[n_ + i] = VarStatus::Basic;
            }
        }
        dse_.assign(m_, 1.0);
        if (!refactor()) return Status::NumericalError;
        compute_primal(lb_, ub_);
        compute_dual();
        int infeas = place_nonbasics(lb_, ub_);
        if (opt_->verbose && crossover_) {
            int pinf = 0; double mx = 0;
            for (int k = 0; k < m_; ++k) {
                const int pv = head_[k];
                const double v = std::max(lb_[pv] - x_[pv], x_[pv] - ub_[pv]);
                if (v > ptol_) { ++pinf; mx = std::max(mx, v); }
            }
            std::printf("[simplex] crossover start: %d primal infeasible basics (max %.2e), %d dual infeasible\n", pinf, mx, infeas);
        }
        if (opt_->verbose) std::printf("[simplex] m %d n %d initial dual infeasibilities %d\n", m_, n_, infeas);
        if (infeas > 0 && crossover_) {
            // near-optimal start: remove the few dual infeasibilities by shifting
            // costs (the primal simplex restores them at the end)
            for (int j = 0; j < N_; ++j) {
                if (status_[j] == VarStatus::Basic || lb_[j] == ub_[j]) continue;
                const bool bad = (d_[j] > dtol_ && !isfin(lb_[j])) ||
                                 (d_[j] < -dtol_ && !isfin(ub_[j]));
                if (!bad) continue;
                cost_[j] -= d_[j];
                d_[j] = 0.0;
                status_[j] = isfin(lb_[j]) ? VarStatus::AtLower
                             : isfin(ub_[j]) ? VarStatus::AtUpper : VarStatus::AtZero;
            }
            infeas = 0;
        }
        if (infeas > 0) {
            // ---- phase 1: artificial boxed bounds --------------------------------
            std::vector<double> lb1(N_), ub1(N_);
            for (int j = 0; j < N_; ++j) {
                const bool lf = isfin(lb_[j]), uf = isfin(ub_[j]);
                if (!lf && !uf) { lb1[j] = -1000; ub1[j] = 1000; }
                else if (lf && !uf) { lb1[j] = 0; ub1[j] = 1; }
                else if (!lf && uf) { lb1[j] = -1; ub1[j] = 0; }
                else { lb1[j] = 0; ub1[j] = 0; }
            }
            // start from a dual feasible placement for the artificial bounds
            for (int j = 0; j < N_; ++j)
                if (status_[j] != VarStatus::Basic)
                    status_[j] = d_[j] >= 0 ? VarStatus::AtLower : VarStatus::AtUpper;
            compute_primal(lb1, ub1);
            Status s1 = dual_loop(lb1, ub1, true);
            if (s1 != Status::Optimal) return s1 == Status::Infeasible ? Status::Unbounded : s1;
            compute_dual();
            if (opt_->verbose)
                std::printf("[simplex] phase 1 finished after %ld iterations\n", iters_);
            // the artificial problem is optimal: is the basis dual feasible for
            // the real bounds?
            int left = place_nonbasics(lb_, ub_);
            if (left > 0) return Status::Unbounded; // dual infeasible (unbounded or infeasible)
        }
        // ---- phase 2 --------------------------------------------------------------------
        compute_primal(lb_, ub_);
        Status s2 = dual_loop(lb_, ub_, false);
        if (s2 != Status::Optimal) return s2;
        // remove perturbations and shifts, then clean up with the primal simplex
        if (opt_->verbose) std::printf("[simplex] dual phase 2 finished after %ld iterations\n", iters_);
        cost_ = cost0_;
        compute_primal(lb_, ub_);
        compute_dual();
        Status s3 = primal_loop(lb_, ub_);
        if (opt_->verbose) std::printf("[simplex] primal cleanup finished, %ld total iterations\n", iters_);
        return s3;
    }

    LpSolution
    extract(Status st, bool light = false)
    {
        if (opt_->verbose) { // consistency of the final point in the scaled space
            std::vector<double> ax(m_, 0.0);
            const auto *rp = A_->row_ptr(); const auto *ci = A_->col_ind(); const auto *va = A_->values();
            for (int i = 0; i < m_; ++i)
                for (auto t = rp[i]; t < rp[i + 1]; ++t) ax[i] += va[t] * x_[ci[t]];
            double eq = 0, bl = 0;
            for (int i = 0; i < m_; ++i) eq = std::max(eq, std::abs(ax[i] - x_[n_ + i]));
            int bj = -1;
            for (int j = 0; j < N_; ++j) {
                const double v = std::max(lb_[j] - x_[j], x_[j] - ub_[j]);
                if (v > bl) { bl = v; bj = j; }
            }
            if (bj >= 0)
                std::printf("[simplex]   worst var %d status %d pos %d x %.6e lb %.3e ub %.3e dse %.3e\n",
                    bj, static_cast<int>(status_[bj]), pos_[bj], x_[bj], lb_[bj], ub_[bj],
                    pos_[bj] >= 0 ? dse_[pos_[bj]] : 0.0);
            std::printf("[simplex] final: max|Ax-w| %.3e  max bound viol %.3e\n", eq, bl);
        }
        std::vector<double> xv(n_), yv(m_);
        for (int j = 0; j < n_; ++j) xv[j] = x_[j];
        for (int i = 0; i < m_; ++i) yv[i] = y_[i];
        unscale_solution(xv, yv, sc_);
        LpSolution s;
        if (light) { // node LPs: no O(nnz) residual pass, just the point and its objective
            s.x = xv;
            s.y = yv;
            double obj = orig_->offset;
            for (int j = 0; j < n_; ++j) obj += orig_->c[j] * xv[j];
            s.primal_objective = s.dual_objective = obj;
        } else {
            s = evaluate_solution(*orig_, xv, yv);
        }
        s.status = st;
        s.iterations = iters_;
        s.seconds = elapsed();
        return s;
    }
};

} // namespace Solver
} // namespace Panini
