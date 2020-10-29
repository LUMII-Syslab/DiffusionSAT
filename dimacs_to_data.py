"""
Customized version of generators available in NeuroSAT publication & code
"""
import os
import pickle

from pysat.formula import CNF
from pysat.solvers import Cadical

from mk_problem import mk_batch_problem


def solve_sat(clauses):
    formula = CNF(from_clauses=clauses)
    with Cadical(bootstrap_with=formula.clauses) as solver:
        is_sat = solver.solve()

    return is_sat


def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1
    header = lines[i].strip().split(" ")
    assert header[0] == "p"
    n_vars = int(header[2])
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i + 1:]]
    return n_vars, iclauses


def dimacs_to_data(dimacs_dir: str, out_dir: str, max_nodes_per_batch: int, one=0, max_dimacs: int = None):
    problems = []
    batches = []
    n_nodes_in_batch = 0

    filenames = os.listdir(dimacs_dir)

    if not (max_dimacs is None):
        filenames = filenames[:max_dimacs]

    # to improve batching
    filenames = sorted(filenames)

    for filename in filenames:
        n_vars, iclauses = parse_dimacs(f"{dimacs_dir}/{filename}")
        n_clauses = len(iclauses)

        n_nodes = 2 * n_vars + n_clauses
        if n_nodes > max_nodes_per_batch:
            continue

        batch_ready = False
        if one and len(problems) > 0:
            batch_ready = True
        elif (not one) and n_nodes_in_batch + n_nodes > max_nodes_per_batch:
            batch_ready = True

        if batch_ready:
            batches.append(mk_batch_problem(problems))
            print(f"batch {len(batches)} done ({len(problems)} problems)...\n")
            del problems[:]
            n_nodes_in_batch = 0

        is_sat = solve_sat(iclauses)

        if is_sat:
            problems.append((filename, n_vars, iclauses, is_sat))
            n_nodes_in_batch += n_nodes

    if len(problems) > 0:
        batches.append(mk_batch_problem(problems))
        print(f"batch {len(batches)} done ({len(problems)} problems)...\n")
        del problems[:]

    # create directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    dimacs_path = dimacs_dir.split("/")
    dimacs_dir = dimacs_path[-1] if dimacs_path[-1] != "" else dimacs_path[-2]
    dataset_filename = f"{out_dir}/data_dir={dimacs_dir}_npb={max_nodes_per_batch}_nb={len(batches)}.pkl"

    print(f"Writing {len(batches)} batches to {dataset_filename}...\n")
    with open(dataset_filename, 'wb') as f_dump:
        pickle.dump(batches, f_dump)
