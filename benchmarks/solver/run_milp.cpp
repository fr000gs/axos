// SPDX-License-Identifier: BSD-3-Clause
//
// Solves MPS files (with integer markers) by branch and bound and prints
//   RESULT milp <name> <ms> <nodes> <status> <objective> <bound> <gap> <lp-iterations>
// Objective and bound are in the file's own sense (a maximization model is
// reported as maximization).
//
//   run_milp [--time s] [--gap g] [--nodes n] [--no-prop] [--no-dive] [-v] file.mps ...
#include "solver/solver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace Panini;
using namespace Panini::Solver;

int
main(int argc, char **argv)
{
    MilpOptions o;
    o.time_limit = 120;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--time" && i + 1 < argc) o.time_limit = atof(argv[++i]);
        else if (a == "--gap" && i + 1 < argc) o.mip_gap = atof(argv[++i]);
        else if (a == "--nodes" && i + 1 < argc) o.node_limit = atol(argv[++i]);
        else if (a == "--no-prop") o.propagate = false;
        else if (a == "--no-dive") o.diving = false;
        else if (a == "--no-presolve") o.presolve = false;
        else if (a == "--no-cuts") o.cuts = false;
        else if (a == "--no-gomory") o.gomory = false;
        else if (a == "-v") o.verbose = true;
        else files.push_back(a);
    }
    for (const auto &f : files) {
        std::string name = f.substr(f.find_last_of('/') + 1);
        name = name.substr(0, name.rfind('.'));
        LpProblem p;
        try {
            p = read_mps_file(f);
        } catch (const std::exception &e) {
            printf("RESULT milp %s 0 0 readerror nan nan nan 0 # %s\n", name.c_str(), e.what());
            continue;
        }
        const double sgn = p.maximize ? -1.0 : 1.0;
        MilpSolution s = solve_milp(p, o);
        printf("RESULT milp %s %.1f %ld %s %.10g %.10g %.2e %ld\n", name.c_str(), s.seconds * 1000,
            s.nodes, to_string(s.status), s.has_solution() ? sgn * s.objective : NAN,
            std::isfinite(s.best_bound) ? sgn * s.best_bound : NAN, s.gap, s.lp_iterations);
        fflush(stdout);
    }
    return 0;
}
