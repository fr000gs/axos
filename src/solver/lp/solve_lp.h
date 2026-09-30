// SPDX-License-Identifier: BSD-3-Clause
//
// solve_lp(): presolve -> LP algorithm (see SolverOptions::method; Auto runs
// several, concurrently by default) -> postsolve.
#pragma once

#include "solver/lp/ipm.h"
#include "solver/lp/pdlp.h"
#include "solver/lp/simplex.h"
#include <mutex>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Panini {
namespace Solver {

// Presolve + solve + postsolve. The returned solution is for `p` as given.
template <template <typename> class Store = Cpu::HostStorage>
LpSolution
solve_lp(const LpProblem &p, const SolverOptions &opt_in = SolverOptions())
{
    SolverOptions opt = opt_in; // the time limit is lowered for the presolve retry
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    };
    // The interior-point method needs presolve (free rows, fixed columns).
    const bool need_presolve = opt.presolve || opt.method != LpMethod::Pdlp ||
                               p.rows() == 0 || p.cols() == 0;
    auto usable = [](const LpProblem &q, const LpSolution &s) {
        if (s.x.size() != q.cols() || s.y.size() != q.rows()) return false;
        for (double v : s.x) if (!std::isfinite(v)) return false;
        for (double v : s.y) if (!std::isfinite(v)) return false;
        return true;
    };
    // `start` (an interior / approximate solution of q) seeds the simplex by crossover
    auto run = [&](const LpProblem &q, LpMethod m, const SolverOptions &o,
                   const LpSolution *start = nullptr) {
        switch (m) {
        case LpMethod::Ipm: {
            Ipm<Store> ipm;
            LpSolution s = ipm.solve(q, o);
            // the normal equations can fail where the augmented system does not
            // (nearly dependent rows): retry with it
            if (s.status == Status::NumericalError && ipm.used_normal() && o.ipm_normal == 0) {
                SolverOptions oa = o;
                oa.ipm_normal = -1;
                oa.time_limit = std::max(0.0, o.time_limit - s.seconds);
                if (oa.time_limit > 0) {
                    const double t0s = s.seconds;
                    s = Ipm<Store>().solve(q, oa);
                    s.seconds += t0s;
                }
            }
            if (o.crossover && s.status == Status::Optimal && usable(q, s)) {
                SolverOptions oc = o;
                oc.time_limit = std::max(1.0, o.time_limit - s.seconds);
                LpSolution c = DualSimplex().solve(q, oc, nullptr, nullptr, &s);
                if (c.status == Status::Optimal) return c;
            }
            return s;
        }
        case LpMethod::Simplex: return DualSimplex().solve(q, o, nullptr, nullptr, start);
        default: return Pdlp<Store>().solve(q, o);
        }
    };
    // Auto. A cheap interior-point factorization (predicted from the symbolic
    // analysis) means the interior-point method goes first, then simplex, then
    // PDLP. Otherwise the dual simplex gets a slice of the predicted interior-
    // point time (its typical win on network / sparse problems), then the
    // interior-point method, then PDLP. The time limit is shared.
    auto run_auto = [&](const LpProblem &q) {
        constexpr double kCheapFlops = 4e9;    // Sum of squared column counts
        constexpr double kFlopsPerSec = 4e10;  // measured multifrontal rate on the CPU
        auto remaining = [&] { return std::max(0.0, opt.time_limit - secs()); };
        auto accept = [&](LpMethod m, const LpSolution &s) {
            // Optimal, or a certificate from a method that can give one
            return s.status == Status::Optimal ||
                   (m != LpMethod::Ipm && (s.status == Status::Infeasible ||
                                           s.status == Status::Unbounded));
        };
        auto attempt = [&](LpMethod m, double budget, LpSolution &best, double flops_limit = 0) {
            SolverOptions o = opt;
            o.time_limit = std::min(budget, remaining());
            o.ipm_max_flops = flops_limit;
            if (o.time_limit <= 0) return false;
            LpSolution s = run(q, m, o);
            if (accept(m, s)) { best = s; return true; }
            if (s.status != Status::NotSolved) best = s;
            else if (best.status == Status::NotSolved) best.factor_flops = s.factor_flops;
            return false;
        };
        LpSolution best;
        if (attempt(LpMethod::Ipm, remaining(), best, kCheapFlops)) return best;
        if (best.status == Status::NotSolved) { // IPM predicted to be expensive
            const double t_ipm = 20.0 * best.factor_flops / kFlopsPerSec;
            if (attempt(LpMethod::Simplex, std::max(2.0, 0.3 * t_ipm), best)) return best;
            if (attempt(LpMethod::Ipm, 0.6 * remaining(), best)) return best;
            attempt(LpMethod::Pdlp, remaining(), best);
            return best;
        }
        if (attempt(LpMethod::Simplex, 0.5 * remaining(), best)) return best;
        attempt(LpMethod::Pdlp, remaining(), best);
        return best;
    };
    // Concurrent Auto: the dual simplex on one CPU thread; on the other, the
    // interior-point method (unless its predicted CPU factorization cost exceeds
    // the time limit) and then PDLP, on the storage's device. The first result
    // that is optimal or a certificate stops the other thread.
    auto run_concurrent = [&](const LpProblem &q) {
        // sustained factorization rate: multifrontal on the CPU, cuDSS (FP64 on
        // a consumer GPU) on CUDA; both measured on this project's benchmarks
        constexpr bool on_cpu = std::is_same_v<Store<double>, Cpu::HostStorage<double>>;
        constexpr double kFlopsPerSec = on_cpu ? 4e10 : 1e11;
        auto remaining = [&] { return std::max(0.0, opt.time_limit - secs()); };
        auto accept = [&](LpMethod m, const LpSolution &s) {
            return s.status == Status::Optimal ||
                   (m != LpMethod::Ipm && (s.status == Status::Infeasible ||
                                           s.status == Status::Unbounded));
        };
        StopFlag stop;
        stop.parent = opt.interrupt;
        std::mutex mu;
        bool have = false;
        LpSolution winner, fallback;
        auto offer = [&](LpMethod m, LpSolution &&s) {
            std::lock_guard<std::mutex> lk(mu);
            if (have) return;
            if (accept(m, s)) {
                winner = std::move(s);
                have = true;
                stop.set();
            } else if (s.status != Status::NotSolved && s.status != Status::Interrupted &&
                       (fallback.status == Status::NotSolved || m == LpMethod::Simplex)) {
                fallback = std::move(s); // most informative non-accepted result
            }
        };
        SolverOptions base = opt;
        base.interrupt = &stop;
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#endif
        std::exception_ptr err_s, err_d;
        std::thread simplex_thread([&] {
            try {
#ifdef _OPENMP
                omp_set_num_threads(1); // the other thread has the remaining cores
#endif
                SolverOptions o = base;
                o.time_limit = remaining();
                offer(LpMethod::Simplex, run(q, LpMethod::Simplex, o));
            } catch (...) { err_s = std::current_exception(); }
        });
        std::thread device_thread([&] {
            try {
#ifdef _OPENMP
                // leave a core for the simplex thread
                if constexpr (std::is_same_v<Store<double>, Cpu::HostStorage<double>>)
                    omp_set_num_threads(std::max(1, nthreads - 1));
#endif
                SolverOptions o = base;
                o.time_limit = remaining();
                // 0. Opt-in (SolverOptions::pdlp_first), large problems: PDLP gets a short slice first. The flops estimate
                //    misses the ordering / analysis cost of a big KKT system
                //    (savsched1: 'cheap' at 4e9 flops, yet 23 s against 0.9 s for PDLP).
                if ((opt.pdlp_first > 0 || (opt.pdlp_first < 0 && !on_cpu)) && q.A.nnz() >= 50000) {
                    o.time_limit = std::min(remaining(), std::min(8.0, std::max(2.0, 0.05 * opt.time_limit)));
                    if (o.time_limit > 0) offer(LpMethod::Pdlp, run(q, LpMethod::Pdlp, o));
                    if (stop.get()) return;
                    o.time_limit = remaining();
                }
                // 1. The interior-point method, if one factorization is cheap
                //    (about 0.2-0.5 s): it then finishes in seconds.
                constexpr double kCheapFlops = 2e10;
                o.ipm_max_flops = kCheapFlops;
                LpSolution first = run(q, LpMethod::Ipm, o);
                offer(LpMethod::Ipm, LpSolution(first));
                if (stop.get()) return;
                // 2. Otherwise PDLP gets a slice first: it is cheap per iteration
                //    and often converges long before ~25 expensive factorizations
                //    would (ex10: 1.7 s vs 41 s on the GPU).
                if (first.status == Status::NotSolved) {
                    const double est_ipm = 25.0 * first.factor_flops / kFlopsPerSec;
                    o.ipm_max_flops = 0;
                    o.time_limit = std::min(remaining(), std::max(2.0, 0.3 * est_ipm));
                    if (o.time_limit > 0) offer(LpMethod::Pdlp, run(q, LpMethod::Pdlp, o));
                    if (stop.get()) return;
                    // 3. The interior-point method, unless it is predicted to take
                    //    more than the time that is left
                    o = base;
                    o.time_limit = remaining();
                    o.ipm_max_flops = remaining() * kFlopsPerSec / 20.0;
                    if (o.time_limit > 0) offer(LpMethod::Ipm, run(q, LpMethod::Ipm, o));
                    if (stop.get()) return;
                }
                // 4. PDLP for whatever time remains
                o = base;
                o.time_limit = remaining();
                if (o.time_limit > 0) offer(LpMethod::Pdlp, run(q, LpMethod::Pdlp, o));
            } catch (...) { err_d = std::current_exception(); }
        });
        simplex_thread.join();
        device_thread.join();
        if (have) return winner;
        if (err_s) std::rethrow_exception(err_s);
        if (err_d) std::rethrow_exception(err_d);
        if (fallback.status == Status::NotSolved && opt.interrupted())
            fallback.status = Status::Interrupted;
        return fallback;
    };
    auto auto_or_run = [&](const LpProblem &q) {
        if (opt.method != LpMethod::Auto) {
            SolverOptions o = opt;
            o.time_limit = std::max(0.0, opt.time_limit - secs()); // a retry gets what is left
            return run(q, opt.method, o);
        }
        return opt.concurrent ? run_concurrent(q) : run_auto(q);
    };
    if (!need_presolve) return auto_or_run(p);

    // Optimal as reported for the reduced problem must also hold on the original
    // one: the tolerances are checked on the postsolved (x, y).
    double nb_orig = 0, nc_orig = 0;
    for (size_t i = 0; i < p.rows(); ++i) {
        if (std::isfinite(p.row_lb[i])) nb_orig += p.row_lb[i] * p.row_lb[i];
        if (std::isfinite(p.row_ub[i])) nb_orig += p.row_ub[i] * p.row_ub[i];
    }
    for (double v : p.c) nc_orig += v * v;
    nb_orig = std::sqrt(nb_orig);
    nc_orig = std::sqrt(nc_orig);
    constexpr double kSlack = 10.0; // postsolve adds round-off
    auto primal_holds = [&](const LpSolution &s) {
        return s.primal_residual <= kSlack * opt_in.eps_primal * (1 + nb_orig);
    };
    auto holds = [&](const LpSolution &s) {
        return primal_holds(s) &&
               s.dual_residual <= kSlack * opt_in.eps_dual * (1 + nc_orig) &&
               s.gap <= kSlack * opt_in.eps_gap * (1 + std::abs(s.primal_objective) + std::abs(s.dual_objective)) &&
               s.error_bound <= 100 * opt_in.eps_gap * (1 + std::abs(s.primal_objective));
    };
    // level 2: all reductions; level 1: the conservative first-generation set
    auto make_presolve_options = [](int level) {
        PresolveOptions o;
        if (level < 2) {
            o.activity_rows = o.dual_fixing = o.singleton_cols = false;
            o.doubleton_equations = o.aggregate = o.parallel_cols = false;
        }
        return o;
    };
    auto solve_with = [&](int level, LpSolution &out) { // false: presolve alone decided
        Presolve pre(p, make_presolve_options(level));
        if (pre.status() != Status::NotSolved) {
            out = LpSolution();
            out.status = pre.status();
            out.x.assign(p.cols(), 0.0);
            out.y.assign(p.rows(), 0.0);
            out.seconds = secs();
            return false;
        }
        const LpProblem &r = pre.reduced();
        LpSolution rs;
        if (r.rows() > 0 && r.cols() > 0) rs = auto_or_run(r);
        else rs.status = Status::Optimal; // nothing left to optimize
        out = pre.postsolve(rs);
        out.seconds = secs();
        return true;
    };
    LpSolution s;
    if (!solve_with(2, s)) return s;
    if (s.status != Status::Optimal || holds(s)) return s;

    // Reported optimal in the presolved space, but the postsolved point fails the
    // tolerances on the original problem. Either a reduction was wrong (the
    // conservative reductions then give the right answer) or the duals are just
    // ill-conditioned (they do not). Retry within a bounded time; without a
    // better result, keep the first one, flagged.
    LpSolution first = s;
    if (opt_in.verbose)
        std::printf("[solve_lp] postsolved solution fails the tolerances (pres %.2e dres %.2e "
                    "gap %.2e): retrying with basic presolve\n",
            first.primal_residual, first.dual_residual, first.gap);
    const double used = secs();
    const double retry_budget = std::max(10.0, used);
    opt.time_limit = std::min(opt_in.time_limit, used + retry_budget);
    LpSolution second;
    if (opt.time_limit > used && solve_with(1, second) && second.status == Status::Optimal &&
        holds(second))
        return second;
    if (primal_holds(first)) {
        first.duals_verified = false;
        first.seconds = secs();
        return first;
    }
    first.status = Status::NumericalError;
    first.seconds = secs();
    return first;
}

} // namespace Solver
} // namespace Panini
