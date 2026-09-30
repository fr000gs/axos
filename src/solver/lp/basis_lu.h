// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse LU of a simplex basis with product-form (eta) updates.
//
//   factor()   Gilbert-Peierls left-looking LU with threshold partial
//              pivoting: columns are ordered by increasing nonzero count
//              (singletons, i.e. slacks, go first) and among acceptable pivots
//              the row with the fewest original nonzeros is preferred.
//              Singular columns are reported so the caller can replace them
//              by slack columns.
//   ftran()    solves B x = b     (b indexed by row, x by basis position)
//   btran()    solves B^T y = c   (c indexed by basis position, y by row)
//   update()   replaces the column at basis position r (eta file), so
//              B_new^{-1} = E B_old^{-1}
//
// All vectors are dense (length m); the triangular solves cost
// O(nnz(L) + nnz(U) + nnz(etas)).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Panini {
namespace Solver {

// Sparse work vector: dense values plus the list of positions that may be
// nonzero (each position appears at most once).
struct SVec {
    std::vector<double> v;
    std::vector<int> idx;
    std::vector<char> flag;

    void init(int n) { v.assign(n, 0.0); flag.assign(n, 0); idx.clear(); }
    int size() const { return static_cast<int>(v.size()); }
    void clear() {
        for (int i : idx) { v[i] = 0.0; flag[i] = 0; }
        idx.clear();
    }
    // v[i] += a, tracking the pattern
    void add(int i, double a) {
        if (!flag[i]) { flag[i] = 1; idx.push_back(i); }
        v[i] += a;
    }
    void set(int i, double a) {
        if (!flag[i]) { flag[i] = 1; idx.push_back(i); }
        v[i] = a;
    }
    // rebuild the pattern from the dense values
    void rebuild_pattern() {
        for (int i : idx) flag[i] = 0;
        idx.clear();
        for (int i = 0; i < size(); ++i)
            if (v[i] != 0.0) { flag[i] = 1; idx.push_back(i); }
    }
};

class BasisFactor {
  public:
    // A sparse column: parallel arrays of row indices and values.
    struct Column {
        const int *idx;
        const double *val;
        int nnz;
    };

    // Factor the m x m matrix whose k-th column is cols[k]. Returns the number
    // of singular pivots; for each, singular_pos[i] is the basis position of the
    // dependent column and free_row[i] a row that was left without a pivot
    // (the caller substitutes the slack of that row).
    //
    // Right-looking Markowitz elimination on a dynamic sparse active matrix
    // (columns hold values, rows hold patterns). Pivots minimize
    // (row count - 1) (column count - 1) among entries that pass the threshold
    // test |a| >= kThresh * (largest |a| in the column); the search examines
    // columns and rows in order of increasing count and stops early (Markowitz
    // / Suhl). Singletons cost 0, so triangular parts of the basis (slacks,
    // network bases) are eliminated first and without fill.
    int
    factor(int m, const std::vector<Column> &cols, std::vector<int> &singular_pos,
        std::vector<int> &free_row)
    {
        m_ = m;
        n_updates_ = 0;
        eta_ptr_.assign(1, 0);
        eta_idx_.clear();
        eta_val_.clear();
        eta_pos_.clear();
        eta_diag_.clear();
        singular_pos.clear();
        free_row.clear();

        constexpr double kThresh = 0.1, kTiny = 1e-11, kDrop = 1e-14;
        constexpr int kSearch = 4; // candidate columns/rows examined per pivot

        // ---- active matrix: pooled columns (values) and rows (patterns) ------
        size_t nnz0 = 0;
        for (int j = 0; j < m; ++j) nnz0 += cols[j].nnz;
        const size_t cap0 = 2 * nnz0 + 4 * static_cast<size_t>(m) + 16;
        cidx_.resize(cap0); cval_.resize(cap0); ridx_.resize(cap0);
        cstart_.resize(m); clen_.resize(m); ccap_.resize(m);
        rstart_.resize(m); rlen_.assign(m, 0); rcap_.resize(m);
        size_t cend = 0, rend = 0;
        for (int j = 0; j < m; ++j) {
            cstart_[j] = cend;
            clen_[j] = 0;
            ccap_[j] = cols[j].nnz + 2;
            for (int t = 0; t < cols[j].nnz; ++t) {
                if (cols[j].val[t] == 0.0) continue;
                cidx_[cend + clen_[j]] = cols[j].idx[t];
                cval_[cend + clen_[j]] = cols[j].val[t];
                ++clen_[j];
                ++rlen_[cols[j].idx[t]];
            }
            cend += ccap_[j];
        }
        for (int i = 0; i < m; ++i) {
            rstart_[i] = rend;
            rcap_[i] = rlen_[i] + 2;
            rend += rcap_[i];
            rlen_[i] = 0;
        }
        for (int j = 0; j < m; ++j)
            for (int t = 0; t < clen_[j]; ++t) {
                const int i = cidx_[cstart_[j] + t];
                ridx_[rstart_[i] + rlen_[i]++] = j;
            }
        // append with relocation to the end of the pool when full
        auto col_push = [&](int j, int i, double v) {
            if (clen_[j] == ccap_[j]) {
                const size_t nc = 2 * static_cast<size_t>(ccap_[j]) + 4;
                if (cend + nc > cidx_.size()) { cidx_.resize(2 * (cend + nc)); cval_.resize(2 * (cend + nc)); }
                std::copy(cidx_.begin() + cstart_[j], cidx_.begin() + cstart_[j] + clen_[j], cidx_.begin() + cend);
                std::copy(cval_.begin() + cstart_[j], cval_.begin() + cstart_[j] + clen_[j], cval_.begin() + cend);
                cstart_[j] = cend;
                ccap_[j] = static_cast<int>(nc);
                cend += nc;
            }
            cidx_[cstart_[j] + clen_[j]] = i;
            cval_[cstart_[j] + clen_[j]] = v;
            ++clen_[j];
        };
        auto row_push = [&](int i, int j) {
            if (rlen_[i] == rcap_[i]) {
                const size_t nc = 2 * static_cast<size_t>(rcap_[i]) + 4;
                if (rend + nc > ridx_.size()) ridx_.resize(2 * (rend + nc));
                std::copy(ridx_.begin() + rstart_[i], ridx_.begin() + rstart_[i] + rlen_[i], ridx_.begin() + rend);
                rstart_[i] = rend;
                rcap_[i] = static_cast<int>(nc);
                rend += nc;
            }
            ridx_[rstart_[i] + rlen_[i]++] = j;
        };
        auto row_remove = [&](int i, int j) { // remove column j from row i's pattern
            int *r = ridx_.data() + rstart_[i];
            for (int t = 0; t < rlen_[i]; ++t)
                if (r[t] == j) { r[t] = r[--rlen_[i]]; return; }
        };

        // ---- count buckets (doubly linked lists) ------------------------------
        chead_.assign(m + 1, -1); rhead_.assign(m + 1, -1);
        cnext_.assign(m, -1); cprev_.assign(m, -1); rnext_.assign(m, -1); rprev_.assign(m, -1);
        std::vector<char> cin(m, 0), rin(m, 0); // in a bucket (= active)
        auto cins = [&](int j) {
            const int c = clen_[j];
            cnext_[j] = chead_[c]; cprev_[j] = -1;
            if (chead_[c] >= 0) cprev_[chead_[c]] = j;
            chead_[c] = j; cin[j] = 1;
        };
        auto cdel = [&](int j) {
            if (!cin[j]) return;
            const int c = clen_[j];
            if (cprev_[j] >= 0) cnext_[cprev_[j]] = cnext_[j]; else chead_[c] = cnext_[j];
            if (cnext_[j] >= 0) cprev_[cnext_[j]] = cprev_[j];
            cin[j] = 0;
        };
        auto rins = [&](int i) {
            const int c = rlen_[i];
            rnext_[i] = rhead_[c]; rprev_[i] = -1;
            if (rhead_[c] >= 0) rprev_[rhead_[c]] = i;
            rhead_[c] = i; rin[i] = 1;
        };
        auto rdel = [&](int i) {
            if (!rin[i]) return;
            const int c = rlen_[i];
            if (rprev_[i] >= 0) rnext_[rprev_[i]] = rnext_[i]; else rhead_[c] = rnext_[i];
            if (rnext_[i] >= 0) rprev_[rnext_[i]] = rprev_[i];
            rin[i] = 0;
        };
        std::vector<char> col_dead(m, 0); // singular columns
        for (int j = 0; j < m; ++j) {
            if (clen_[j] == 0) col_dead[j] = 1; else cins(j);
        }
        for (int i = 0; i < m; ++i)
            if (rlen_[i] > 0) rins(i);

        auto colmax = [&](int j) {
            double mx = 0;
            const double *v = cval_.data() + cstart_[j];
            for (int t = 0; t < clen_[j]; ++t) mx = std::max(mx, std::abs(v[t]));
            return mx;
        };
        auto find_in_col = [&](int j, int i) { // position of row i in column j
            const int *ix = cidx_.data() + cstart_[j];
            for (int t = 0; t < clen_[j]; ++t)
                if (ix[t] == i) return t;
            return -1;
        };
        // a column whose entries are all negligible is singular: drop it
        auto kill_col = [&](int j) {
            cdel(j);
            for (int t = 0; t < clen_[j]; ++t) {
                const int i = cidx_[cstart_[j] + t];
                rdel(i);
                row_remove(i, j);
                if (rlen_[i] > 0) rins(i);
            }
            clen_[j] = 0;
            col_dead[j] = 1;
        };

        // ---- outputs (triplets, converted at the end) ------------------------
        q_.assign(m, -1);
        prow_.assign(m, -1);
        udiag_.assign(m, 1.0);
        std::vector<int> pinv(m, -1), cstep(m, -1);
        std::vector<int> lrow, lstep;   // L entries: (original row, step)
        std::vector<double> lval;
        std::vector<int> urowstep, ucol; // U entries: (step of the pivot row, basis column)
        std::vector<double> uval;
        std::vector<int> wpos(m, -1);    // row -> position in the scattered column
        std::vector<int> inl(m, -1);     // row is in the current L column (stamp k)
        std::vector<int> lcol_i;
        std::vector<double> lcol_v;
        std::vector<int> urow_j;
        std::vector<double> urow_v;

        int k = 0;
        while (k < m) {
            // ---- pivot search ------------------------------------------------
            int pr = -1, pc = -1;
            double best = 1e300, pv = 0;
            int searched = 0;
            for (int cnt = 1; cnt <= m; ++cnt) {
                for (int j = chead_[cnt]; j >= 0;) {
                    const int jn = cnext_[j];
                    const double mx = colmax(j);
                    if (mx < kTiny) { kill_col(j); j = jn; continue; }
                    const int *ix = cidx_.data() + cstart_[j];
                    const double *v = cval_.data() + cstart_[j];
                    for (int t = 0; t < clen_[j]; ++t) {
                        if (std::abs(v[t]) < kThresh * mx) continue;
                        const double cost = static_cast<double>(rlen_[ix[t]] - 1) * (cnt - 1);
                        if (cost < best || (cost == best && std::abs(v[t]) > std::abs(pv))) {
                            best = cost; pr = ix[t]; pc = j; pv = v[t];
                        }
                    }
                    if (++searched >= kSearch && pc >= 0) break;
                    j = jn;
                }
                if (pc >= 0 && (best <= static_cast<double>(cnt - 1) * (cnt - 1) || searched >= kSearch)) break;
                for (int i = rhead_[cnt]; i >= 0; i = rnext_[i]) {
                    const int *rj = ridx_.data() + rstart_[i];
                    for (int t = 0; t < rlen_[i]; ++t) {
                        const int j = rj[t];
                        const int pos = find_in_col(j, i);
                        if (pos < 0) continue;
                        const double a = cval_[cstart_[j] + pos];
                        const double cost = static_cast<double>(cnt - 1) * (clen_[j] - 1);
                        if (cost >= best) continue;
                        if (std::abs(a) < kThresh * colmax(j) || std::abs(a) < kTiny) continue;
                        best = cost; pr = i; pc = j; pv = a;
                    }
                    if (++searched >= kSearch && pc >= 0) break;
                }
                if (pc >= 0 && (best <= static_cast<double>(cnt) * cnt || searched >= kSearch)) break;
            }
            if (pc < 0) break; // the remaining columns are singular

            // ---- eliminate (pr, pc) -------------------------------------------
            q_[k] = pc; prow_[k] = pr; udiag_[k] = pv;
            pinv[pr] = k; cstep[pc] = k;
            cdel(pc);
            rdel(pr);
            // L column: the other rows of column pc
            lcol_i.clear(); lcol_v.clear();
            for (int t = 0; t < clen_[pc]; ++t) {
                const int i = cidx_[cstart_[pc] + t];
                if (i == pr) continue;
                const double l = cval_[cstart_[pc] + t] / pv;
                lcol_i.push_back(i); lcol_v.push_back(l);
                lrow.push_back(i); lstep.push_back(k); lval.push_back(l);
            }
            // rows of column pc lose it
            for (int i : lcol_i) { rdel(i); row_remove(i, pc); inl[i] = k; }
            clen_[pc] = 0;
            // U row: the other columns of row pr (their entry in row pr is removed)
            urow_j.clear(); urow_v.clear();
            {
                const int *rj = ridx_.data() + rstart_[pr];
                for (int t = 0; t < rlen_[pr]; ++t) {
                    const int j = rj[t];
                    if (j == pc) continue;
                    const int pos = find_in_col(j, pr);
                    if (pos < 0) continue;
                    const double a = cval_[cstart_[j] + pos];
                    cidx_[cstart_[j] + pos] = cidx_[cstart_[j] + clen_[j] - 1];
                    cval_[cstart_[j] + pos] = cval_[cstart_[j] + clen_[j] - 1];
                    cdel(j);
                    --clen_[j];
                    urow_j.push_back(j); urow_v.push_back(a);
                    urowstep.push_back(k); ucol.push_back(j); uval.push_back(a);
                }
                rlen_[pr] = 0;
            }
            // Schur update: column j -= a_{pr,j} * lcol
            if (!lcol_i.empty()) {
                for (size_t u = 0; u < urow_j.size(); ++u) {
                    const int j = urow_j[u];
                    const double a = urow_v[u];
                    for (int t = 0; t < clen_[j]; ++t) wpos[cidx_[cstart_[j] + t]] = t;
                    for (size_t l = 0; l < lcol_i.size(); ++l) {
                        const int i = lcol_i[l];
                        const double d = -lcol_v[l] * a;
                        if (wpos[i] >= 0) {
                            cval_[cstart_[j] + wpos[i]] += d;
                        } else { // fill-in
                            col_push(j, i, d);
                            wpos[i] = clen_[j] - 1;
                            row_push(i, j);
                        }
                    }
                    // clear the map and drop updated entries that cancelled (only
                    // rows of the L column: they are out of their buckets)
                    for (int t = 0; t < clen_[j];) {
                        const int i = cidx_[cstart_[j] + t];
                        wpos[i] = -1;
                        if (inl[i] == k && std::abs(cval_[cstart_[j] + t]) < kDrop) {
                            row_remove(i, j);
                            cidx_[cstart_[j] + t] = cidx_[cstart_[j] + clen_[j] - 1];
                            cval_[cstart_[j] + t] = cval_[cstart_[j] + clen_[j] - 1];
                            --clen_[j];
                            continue;
                        }
                        ++t;
                    }
                }
            }
            // back into the buckets with their new counts
            for (int j : urow_j) {
                if (clen_[j] == 0) col_dead[j] = 1; else cins(j);
            }
            for (int i : lcol_i)
                if (rlen_[i] > 0) rins(i);
            ++k;
        }
        const int nsing = m - k;

        // ---- singular columns: remaining steps without a pivot row ----------
        for (int j = 0; j < m; ++j)
            if (cstep[j] < 0) {
                singular_pos.push_back(j);
                q_[k] = j; cstep[j] = k; prow_[k] = -1; udiag_[k] = 1.0;
                ++k;
            }
        for (int r = 0; r < m && free_row.size() < singular_pos.size(); ++r)
            if (pinv[r] < 0) free_row.push_back(r);

        // ---- L by step (entries in unpivoted rows only occur when singular) --
        lp_.assign(m + 1, 0);
        for (size_t t = 0; t < lrow.size(); ++t)
            if (pinv[lrow[t]] >= 0) ++lp_[lstep[t] + 1];
        for (int s0 = 0; s0 < m; ++s0) lp_[s0 + 1] += lp_[s0];
        li_.resize(lp_[m]); lx_.resize(lp_[m]);
        {
            std::vector<int> nx(lp_.begin(), lp_.end() - 1);
            for (size_t t = 0; t < lrow.size(); ++t) {
                const int st = pinv[lrow[t]];
                if (st < 0) continue;
                const int q = nx[lstep[t]]++;
                li_[q] = st;
                lx_[q] = lval[t];
            }
        }
        // ---- U by step of the column: entries at earlier steps -------------
        up_.assign(m + 1, 0);
        for (size_t t = 0; t < ucol.size(); ++t) ++up_[cstep[ucol[t]] + 1];
        for (int s0 = 0; s0 < m; ++s0) up_[s0 + 1] += up_[s0];
        ui_.resize(up_[m]); ux_.resize(up_[m]);
        {
            std::vector<int> nx(up_.begin(), up_.end() - 1);
            for (size_t t = 0; t < ucol.size(); ++t) {
                const int q = nx[cstep[ucol[t]]]++;
                ui_[q] = urowstep[t];
                ux_[q] = uval[t];
            }
        }
        singular_ = nsing;
        pinv_ = pinv;
        qinv_.assign(m, 0);
        for (int s0 = 0; s0 < m; ++s0) qinv_[q_[s0]] = s0;
        mark_.assign(m, 0);
        stamp_ = 0;
        w_.assign(m, 0.0);
        // row-wise copies of L and U so that BTRAN can skip zero entries
        transpose(lp_, li_, lx_, lrp_, lci_, lrx_);
        transpose(up_, ui_, ux_, urp_, uci_, urx_);
        return nsing;
    }

    // Solve B x = b in place: on input b is indexed by row, on output by basis
    // position. `work` must have m entries.
    void
    ftran(std::vector<double> &b) const
    {
        const int m = m_;
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = prow_[k] >= 0 ? b[prow_[k]] : 0.0;
        for (int k = 0; k < m; ++k) { // L
            double yk = w_[k];
            if (yk == 0.0) continue;
            for (int p = lp_[k]; p < lp_[k + 1]; ++p) w_[li_[p]] -= lx_[p] * yk;
        }
        for (int k = m - 1; k >= 0; --k) { // U (upper triangular in step order)
            double zk = w_[k] / udiag_[k];
            w_[k] = zk;
            if (zk == 0.0) continue;
            for (int p = up_[k]; p < up_[k + 1]; ++p) w_[ui_[p]] -= ux_[p] * zk;
        }
        for (int k = 0; k < m; ++k) b[q_[k]] = w_[k];
        std::fill(w_.begin(), w_.end(), 0.0); // the sparse solves expect w_ == 0
        // eta file, oldest first
        for (int e = 0; e < n_updates_; ++e) {
            const int r = eta_pos_[e];
            const double tr = b[r];
            if (tr == 0.0) continue;
            b[r] = tr * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                b[eta_idx_[p]] += eta_val_[p] * tr;
        }
    }

    // Solve B^T y = c: c indexed by basis position, y by row.
    void
    btran(std::vector<double> &c) const
    {
        const int m = m_;
        for (int e = n_updates_ - 1; e >= 0; --e) { // eta file, newest first
            const int r = eta_pos_[e];
            double s = c[r] * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                s += eta_val_[p] * c[eta_idx_[p]];
            c[r] = s;
        }
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = c[q_[k]];
        for (int i = 0; i < m; ++i) { // U^T z = w, row-wise sweep
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        for (int i = m - 1; i >= 0; --i) { // L^T v = z, row-wise sweep
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        for (int k = 0; k < m; ++k)
            if (prow_[k] >= 0) c[prow_[k]] = w_[k];
        std::fill(w_.begin(), w_.end(), 0.0);
    }

    // ---- hypersparse solves ---------------------------------------------------
    // Same operations as ftran/btran, on a sparse vector: the triangular solves
    // visit only the nonzero pattern of the result (symbolic reach by DFS), and
    // fall back to the dense loops when the vector is not sparse.

    // Solve B x = b: b indexed by row on input, by basis position on output.
    void
    ftran(SVec &b) const
    {
        const int m = m_;
        if (static_cast<int>(b.idx.size()) > 0.1 * m) {
            ftran(b.v);
            b.rebuild_pattern();
            return;
        }
        // rows -> steps
        starts_.clear();
        for (int i : b.idx) {
            const int k = pinv_[i];
            const double val = b.v[i];
            b.v[i] = 0.0; b.flag[i] = 0;
            if (k < 0 || val == 0.0) continue;
            w_[k] = val;
            starts_.push_back(k);
        }
        b.idx.clear();
        reach(lp_, li_, starts_, order_);       // L: edges step -> larger steps
        for (int t = static_cast<int>(order_.size()) - 1; t >= 0; --t) {
            const int k = order_[t];
            const double yk = w_[k];
            if (yk == 0.0) continue;
            for (int p = lp_[k]; p < lp_[k + 1]; ++p) w_[li_[p]] -= lx_[p] * yk;
        }
        starts_.assign(order_.begin(), order_.end());
        reach(up_, ui_, starts_, order2_);      // U: edges step -> smaller steps
        for (int t = static_cast<int>(order2_.size()) - 1; t >= 0; --t) {
            const int k = order2_[t];
            const double zk = w_[k] / udiag_[k];
            w_[k] = zk;
            if (zk == 0.0) continue;
            for (int p = up_[k]; p < up_[k + 1]; ++p) w_[ui_[p]] -= ux_[p] * zk;
        }
        for (int k : order2_) {
            const double val = w_[k];
            w_[k] = 0.0;
            if (val != 0.0) b.set(q_[k], val);
        }
        for (int e = 0; e < n_updates_; ++e) { // eta file, oldest first
            const int r = eta_pos_[e];
            const double tr = b.v[r];
            if (tr == 0.0) continue;
            b.v[r] = tr * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                b.add(eta_idx_[p], eta_val_[p] * tr);
        }
    }

    // Solve B^T y = c: c indexed by basis position on input, by row on output.
    void
    btran(SVec &c) const
    {
        const int m = m_;
        if (static_cast<int>(c.idx.size()) > 0.1 * m && n_updates_ == 0) {
            btran(c.v);
            c.rebuild_pattern();
            return;
        }
        for (int e = n_updates_ - 1; e >= 0; --e) { // eta file, newest first
            const int r = eta_pos_[e];
            double sum = c.v[r] * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                sum += eta_val_[p] * c.v[eta_idx_[p]];
            if (sum != 0.0 || c.v[r] != 0.0) c.set(r, sum);
        }
        if (static_cast<int>(c.idx.size()) > 0.1 * m) {
            btran_lu_dense(c);
            return;
        }
        starts_.clear();
        for (int pos : c.idx) {
            const double val = c.v[pos];
            c.v[pos] = 0.0; c.flag[pos] = 0;
            if (val == 0.0) continue;
            const int k = qinv_[pos];
            w_[k] = val;
            starts_.push_back(k);
        }
        c.idx.clear();
        reach(urp_, uci_, starts_, order_);     // U^T: row-wise U, edges to larger steps
        for (int t = static_cast<int>(order_.size()) - 1; t >= 0; --t) {
            const int i = order_[t];
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        starts_.assign(order_.begin(), order_.end());
        reach(lrp_, lci_, starts_, order2_);    // L^T: row-wise L, edges to smaller steps
        for (int t = static_cast<int>(order2_.size()) - 1; t >= 0; --t) {
            const int i = order2_[t];
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        for (int k : order2_) {
            const double val = w_[k];
            w_[k] = 0.0;
            if (val != 0.0 && prow_[k] >= 0) c.set(prow_[k], val);
        }
    }

    // Column at basis position r is replaced; alpha = B_old^{-1} a_q (indexed
    // by basis position). Returns false if the pivot is unusably small.
    bool
    update(int r, const std::vector<double> &alpha)
    {
        const double ar = alpha[r];
        if (std::abs(ar) < 1e-11) return false;
        eta_pos_.push_back(r);
        eta_diag_.push_back(1.0 / ar);
        for (int i = 0; i < m_; ++i) {
            if (i == r) continue;
            const double v = alpha[i];
            if (v != 0.0 && std::abs(v) > 1e-14) {
                eta_idx_.push_back(i);
                eta_val_.push_back(-v / ar);
            }
        }
        eta_ptr_.push_back(static_cast<int>(eta_idx_.size()));
        ++n_updates_;
        return true;
    }

    // Same as update(r, alpha) with a sparse alpha.
    bool
    update(int r, const SVec &alpha)
    {
        const double ar = alpha.v[r];
        if (std::abs(ar) < 1e-11) return false;
        eta_pos_.push_back(r);
        eta_diag_.push_back(1.0 / ar);
        for (int i : alpha.idx) {
            if (i == r) continue;
            const double v = alpha.v[i];
            if (v != 0.0 && std::abs(v) > 1e-14) {
                eta_idx_.push_back(i);
                eta_val_.push_back(-v / ar);
            }
        }
        eta_ptr_.push_back(static_cast<int>(eta_idx_.size()));
        ++n_updates_;
        return true;
    }

    int updates() const { return n_updates_; }
    size_t lu_nnz() const { return li_.size() + ui_.size() + m_; }
    // true when the eta file has grown larger than the factors it extends
    bool eta_heavy(double factor) const { return eta_idx_.size() > factor * (li_.size() + ui_.size() + m_); }
    size_t nnz() const { return li_.size() + ui_.size() + eta_idx_.size() + m_; }
    int size() const { return m_; }

  private:
    // The L/U triangular part of btran on a dense-ish SVec (no eta file).
    void
    btran_lu_dense(SVec &c) const
    {
        const int m = m_;
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = c.v[q_[k]];
        for (int i = 0; i < m; ++i) {
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        for (int i = m - 1; i >= 0; --i) {
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        c.clear();
        for (int k = 0; k < m; ++k)
            if (prow_[k] >= 0 && w_[k] != 0.0) c.set(prow_[k], w_[k]);
        std::fill(w_.begin(), w_.end(), 0.0);
    }

    // Nodes reachable from `starts` in the graph with adjacency (ptr, idx), in
    // DFS finishing order (so the reverse is a topological order).
    void
    reach(const std::vector<int> &ptr, const std::vector<int> &idx,
        const std::vector<int> &starts, std::vector<int> &out) const
    {
        out.clear();
        if (++stamp_ == 0x7fffffff) { std::fill(mark_.begin(), mark_.end(), 0); stamp_ = 1; }
        for (int s0 : starts) {
            if (mark_[s0] == stamp_) continue;
            mark_[s0] = stamp_;
            stk_.assign(1, s0);
            pst_.assign(1, ptr[s0]);
            while (!stk_.empty()) {
                const int node = stk_.back();
                int &pp = pst_.back();
                bool pushed = false;
                while (pp < ptr[node + 1]) {
                    const int ch = idx[pp++];
                    if (mark_[ch] != stamp_) {
                        mark_[ch] = stamp_;
                        stk_.push_back(ch);
                        pst_.push_back(ptr[ch]);
                        pushed = true;
                        break;
                    }
                }
                if (!pushed) { out.push_back(node); stk_.pop_back(); pst_.pop_back(); }
            }
        }
    }

    // CSR transpose of a column-wise triangular factor (column k holds row
    // indices idx[ptr[k]..ptr[k+1])).
    void
    transpose(const std::vector<int> &ptr, const std::vector<int> &idx,
        const std::vector<double> &val, std::vector<int> &rp, std::vector<int> &ci,
        std::vector<double> &rx) const
    {
        const int m = m_;
        rp.assign(m + 1, 0);
        for (size_t p = 0; p < idx.size(); ++p) ++rp[idx[p] + 1];
        for (int i = 0; i < m; ++i) rp[i + 1] += rp[i];
        ci.resize(idx.size());
        rx.resize(idx.size());
        std::vector<int> pos(rp.begin(), rp.end() - 1);
        for (int k = 0; k < m; ++k)
            for (int p = ptr[k]; p < ptr[k + 1]; ++p) {
                const int q = pos[idx[p]]++;
                ci[q] = k;
                rx[q] = val[p];
            }
    }

    int m_ = 0, n_updates_ = 0, singular_ = 0;
    std::vector<int> q_, prow_;
    std::vector<int> lp_, li_, up_, ui_;
    std::vector<double> lx_, ux_, udiag_;
    std::vector<int> lrp_, lci_, urp_, uci_;
    std::vector<double> lrx_, urx_;
    std::vector<int> eta_ptr_{0}, eta_idx_, eta_pos_;
    std::vector<double> eta_val_, eta_diag_;
    mutable std::vector<double> w_;
    std::vector<int> pinv_, qinv_;
    // factor() workspace, kept between refactorizations
    std::vector<int> cidx_, ridx_, clen_, ccap_, rlen_, rcap_;
    std::vector<size_t> cstart_, rstart_;
    std::vector<double> cval_;
    std::vector<int> chead_, rhead_, cnext_, cprev_, rnext_, rprev_;
    mutable std::vector<int> mark_, starts_, order_, order2_, stk_, pst_;
    mutable int stamp_ = 0;
};

} // namespace Solver
} // namespace Panini
