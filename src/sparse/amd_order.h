// SPDX-License-Identifier: BSD-3-Clause
//
// Approximate minimum degree ordering on the graph of a symmetric sparse
// matrix, written from the published algorithm (Amestoy, Davis, Duff,
// "An approximate minimum degree ordering algorithm", SIMAX 1996):
//
//   * quotient graph: every variable keeps a list of adjacent variables (A_i)
//     and adjacent elements (E_i); an eliminated pivot becomes an element
//     whose variable list L_e replaces the clique it would create
//   * approximate external degree with the |L_e \ L_p| trick
//   * element absorption (aggressive) and mass elimination of variables that
//     become indistinguishable from the pivot
//   * supervariables: variables of L_p with identical adjacency (A_i, E_i) are
//     merged (hashing, then exact comparison) and carry a weight nv; degrees
//     and element sizes are weighted
//   * dense variables (degree > 10 sqrt(n)) are ordered last
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace Panini {
namespace Sparse {

// Graph of the symmetric pattern of the n x n matrix given by CSR (rp, ci);
// both triangles need not be present (the pattern is symmetrized). Returns
// perm with perm[k] = the k-th eliminated variable. `cancel` (optional) is
// polled every 256 pivots; when it returns true an empty vector is returned.
template <typename Idx>
std::vector<Idx>
amd_order(size_t n, const Idx *rp, const Idx *ci,
    const std::function<bool()> &cancel = nullptr)
{
    std::vector<Idx> perm;
    perm.reserve(n);
    if (n == 0) return perm;

    // ---- build symmetric adjacency (no self loops, no duplicates) --------
    std::vector<std::vector<Idx>> A(n), E(n), L(n);
    for (size_t i = 0; i < n; ++i)
        for (Idx k = rp[i]; k < rp[i + 1]; ++k) {
            Idx j = ci[k];
            if (static_cast<size_t>(j) != i) {
                A[i].push_back(j);
                A[j].push_back(static_cast<Idx>(i));
            }
        }
    for (auto &a : A) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    enum : char { Live = 0, Element = 1, Dead = 2, Dense = 3 };
    std::vector<char> state(n, Live);
    std::vector<Idx> deg(n);
    const size_t dense_thresh = std::max<size_t>(
        16, static_cast<size_t>(10.0 * std::sqrt(static_cast<double>(n))));
    std::vector<Idx> dense_vars;
    size_t nlive = 0;
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() > dense_thresh) {
            state[i] = Dense;
            dense_vars.push_back(static_cast<Idx>(i));
        } else {
            deg[i] = static_cast<Idx>(A[i].size());
            ++nlive;
        }
    }
    // Dense variables no longer count as neighbours of the rest.
    if (!dense_vars.empty()) {
        for (size_t i = 0; i < n; ++i) {
            if (state[i] == Dense) { A[i].clear(); continue; }
            auto &a = A[i];
            a.erase(std::remove_if(a.begin(), a.end(),
                        [&](Idx j) { return state[j] == Dense; }),
                a.end());
            deg[i] = static_cast<Idx>(a.size());
        }
    }

    // ---- degree buckets ------------------------------------------------
    std::vector<Idx> head(n + 1, -1), nxt(n, -1), prv(n, -1);
    auto bucket_insert = [&](Idx i) {
        Idx d = deg[i];
        nxt[i] = head[d];
        prv[i] = -1;
        if (head[d] >= 0) prv[head[d]] = i;
        head[d] = i;
    };
    auto bucket_remove = [&](Idx i) {
        Idx d = deg[i];
        if (prv[i] >= 0) nxt[prv[i]] = nxt[i];
        else head[d] = nxt[i];
        if (nxt[i] >= 0) prv[nxt[i]] = prv[i];
    };
    for (size_t i = 0; i < n; ++i)
        if (state[i] == Live) bucket_insert(static_cast<Idx>(i));
    size_t mindeg = 0;

    // supervariable weights and member chains (merged variables are ordered
    // right after their representative)
    std::vector<Idx> nv(n, 1), mnext(n, -1), mlast(n);
    for (size_t i = 0; i < n; ++i) mlast[i] = static_cast<Idx>(i);
    std::vector<long> Lw(n, 0);        // weighted |L_e| of live elements
    auto emit = [&](Idx v) {
        for (Idx u = v; u >= 0; u = mnext[u]) perm.push_back(u);
    };

    std::vector<Idx> mark(n, -1);      // membership in Lp
    std::vector<long> w(n, 0);         // weighted |L_e \ L_p| (per element)
    std::vector<long> wstamp(n, -1);
    std::vector<long> vstamp(n, -1);   // set comparison in supervariable detection
    long vcount = 0;
    std::vector<Idx> lp, mass;
    std::vector<std::pair<size_t, Idx>> hv;
    long remaining = static_cast<long>(nlive); // live weight
    long step = 0;

    while (nlive > 0) {
        while (mindeg <= n && head[mindeg] < 0) ++mindeg;
        if (mindeg > n) break;
        if (cancel && (step & 255) == 0 && cancel()) return {};
        const Idx p = head[mindeg];
        bucket_remove(p);
        emit(p);
        nlive -= nv[p];
        mark[p] = p; // p itself is never a member of L_p
        ++step;

        // ---- form L_p = (A_p U union of L_e, e in E_p) minus p ---------
        lp.clear();
        long lpw = 0;
        for (Idx j : A[p])
            if (state[j] == Live && mark[j] != p) { mark[j] = p; lp.push_back(j); lpw += nv[j]; }
        for (Idx e : E[p]) {
            if (state[e] != Element) continue;
            for (Idx j : L[e])
                if (state[j] == Live && mark[j] != p) {
                    mark[j] = p;
                    lp.push_back(j);
                    lpw += nv[j];
                }
            state[e] = Dead; // absorbed into p
            L[e].clear();
            L[e].shrink_to_fit();
        }
        A[p].clear();
        E[p].clear();
        A[p].shrink_to_fit();

        // ---- |L_e \ L_p| (weighted) for elements touching L_p ------------
        for (Idx i : lp)
            for (Idx e : E[i]) {
                if (state[e] != Element) continue;
                if (wstamp[e] != step) {
                    wstamp[e] = step;
                    w[e] = Lw[e];
                }
                w[e] -= nv[i];
            }

        // ---- update every variable of L_p ------------------------------
        mass.clear();
        hv.clear();
        for (Idx i : lp) {
            bucket_remove(i);
            // prune E_i: drop absorbed elements, aggressive absorption
            auto &Ei = E[i];
            size_t out = 0;
            long ext = 0; // weighted sum of |L_e \ L_p| over surviving elements
            size_t h = static_cast<size_t>(p);
            for (size_t k = 0; k < Ei.size(); ++k) {
                Idx e = Ei[k];
                if (state[e] != Element) continue;
                if (wstamp[e] == step && w[e] <= 0) { // L_e subset of L_p
                    state[e] = Dead;
                    L[e].clear();
                    L[e].shrink_to_fit();
                    continue;
                }
                ext += (wstamp[e] == step) ? w[e] : Lw[e];
                h += static_cast<size_t>(e);
                Ei[out++] = e;
            }
            Ei.resize(out);
            // prune A_i: variables in L_p (or p) are covered by element p
            auto &Ai = A[i];
            out = 0;
            long aw = 0;
            for (size_t k = 0; k < Ai.size(); ++k) {
                Idx j = Ai[k];
                if (state[j] != Live || mark[j] == p) continue;
                aw += nv[j];
                h += static_cast<size_t>(j);
                Ai[out++] = j;
            }
            Ai.resize(out);
            Ei.push_back(p);

            if (Ai.empty() && Ei.size() == 1) {
                // only adjacent to element p: eliminate together with p
                mass.push_back(i);
                continue;
            }
            const long dp = lpw - nv[i];
            long d = aw + dp + ext;
            d = std::min<long>(d, static_cast<long>(deg[i]) + dp);
            d = std::min<long>(d, remaining - nv[p] - nv[i]);
            if (d < 1) d = 1;
            deg[i] = static_cast<Idx>(std::min<long>(d, static_cast<long>(n)));
            hv.emplace_back(h, i);
        }

        // ---- supervariable detection among the updated variables ------
        std::sort(hv.begin(), hv.end());
        for (size_t a0 = 0; a0 < hv.size();) {
            size_t a1 = a0;
            while (a1 < hv.size() && hv[a1].first == hv[a0].first) ++a1;
            for (size_t x = a0; x < a1; ++x) {
                const Idx i = hv[x].second;
                if (state[i] != Live) continue;
                bool marked = false;
                for (size_t y = x + 1; y < a1; ++y) {
                    const Idx j = hv[y].second;
                    if (state[j] != Live) continue;
                    if (A[i].size() != A[j].size() || E[i].size() != E[j].size()) continue;
                    if (!marked) {
                        ++vcount;
                        for (Idx v : A[i]) vstamp[v] = vcount;
                        for (Idx e : E[i]) vstamp[e] = vcount;
                        marked = true;
                    }
                    bool same = true;
                    for (Idx v : A[j]) if (vstamp[v] != vcount) { same = false; break; }
                    if (same)
                        for (Idx e : E[j]) if (vstamp[e] != vcount) { same = false; break; }
                    if (!same) continue;
                    // merge j into i; i's external degree no longer counts j
                    deg[i] = static_cast<Idx>(std::max<long>(1, static_cast<long>(deg[i]) - nv[j]));
                    nv[i] += nv[j];
                    nv[j] = 0;
                    state[j] = Dead;
                    mnext[mlast[i]] = j;
                    mlast[i] = mlast[j];
                    A[j].clear();
                    A[j].shrink_to_fit();
                    E[j].clear();
                    E[j].shrink_to_fit();
                }
            }
            a0 = a1;
        }

        // ---- finalize the new element ---------------------------------
        long massw = 0;
        for (Idx i : mass) {
            state[i] = Dead;
            emit(i);
            nlive -= nv[i];
            massw += nv[i];
            A[i].clear();
            E[i].clear();
        }
        remaining -= nv[p] + massw;
        L[p].clear();
        long lw = 0;
        for (Idx i : lp)
            if (state[i] == Live) { L[p].push_back(i); lw += nv[i]; }
        Lw[p] = lw;
        state[p] = L[p].empty() ? Dead : Element;
        for (Idx i : L[p]) {
            bucket_insert(i);
            if (static_cast<size_t>(deg[i]) < mindeg) mindeg = deg[i];
        }
    }
    for (size_t i = 0; i < n; ++i) // safety net: never lose a variable
        if (state[i] == Live) perm.push_back(static_cast<Idx>(i));
    for (Idx v : dense_vars) perm.push_back(v);
    return perm;
}

} // namespace Sparse
} // namespace Panini
