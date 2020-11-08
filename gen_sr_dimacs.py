"""
Customized version of generators available in NeuroSAT publication & code
"""
import random

import numpy as np
from pysat.solvers import Cadical


def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write(f"p cnf {n_vars} {len(iclauses)}\n")
        for c in iclauses:
            for x in c:
                f.write(f"{x} ")
            f.write("0\n")


def mk_out_filenames(opts, n_vars, t):
    prefix = f"{opts.out_dir}/sr_n={n_vars}_pk2={opts.p_k_2}_pg={opts.p_geo}_t={t}"
    return f"{prefix}_sat=0.dimacs", f"{prefix}_sat=1.dimacs"


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]


def gen_iclause_pair(min_n, max_n, p_k_2, p_geo):
    n = random.randint(min_n, max_n)

    solver = Cadical()
    iclauses = []

    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)

        solver.add_clause(iclause)
        is_sat = solver.solve()

        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0]] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat

# remove duplicate clauses
# todo: remove subsumed clauses - when shorter clause is fully in a longer one, the longer one is redundant
def prune(clauses):
    clauses_pruned = list({tuple(sorted(x)) for x in clauses})
    return clauses_pruned


def gen_sr_dimacs(out_dir: str, n_pairs: int, min_n=40, max_n=40, p_k_2=0.3, p_geo=0.4, print_interval=100):
    for pair in range(n_pairs):
        if pair % print_interval == 0:
            print(f"[{pair}]")

        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(min_n, max_n, p_k_2, p_geo)
        prefix = f"{out_dir}/sr_n={n_vars}_pk2={p_k_2}_pg={p_geo}_t={pair}"

        iclauses.append(iclause_unsat)
        write_dimacs_to(n_vars, prune(iclauses), f"{prefix}_sat=0.dimacs")

        iclauses[-1] = iclause_sat
        write_dimacs_to(n_vars, prune(iclauses), f"{prefix}_sat=1.dimacs")
