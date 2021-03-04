import subprocess
from pathlib import Path
from typing import Tuple

from utils.iterable import elements_to_str


def remove_unused_vars(nvars, clauses):
    used_vars = set()
    n = 0
    max_v = 0
    for clause in clauses:
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v > max_v:
                max_v = v
            if v not in used_vars:
                used_vars.add(v)
                n += 1
    if n == nvars and max_v == n:
        return nvars, clauses  # do not change since all the variables are used
    # otherwise not all variables are used (or the wrong number specified)

    n = 0
    d = {}
    new_clauses = []
    for clause in clauses:
        new_clause = []
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v in d:
                new_v = d[v]
            else:
                n += 1
                new_v = n
                d[v] = new_v
            if lit > 0:
                new_clause.append(new_v)
            else:
                new_clause.append(-new_v)
        new_clauses.append(new_clause)

    return n, new_clauses


def remove_useless_clauses(clauses):
    """ Removes clauses and variables if variable only appears in clause with single element
    """
    var_n = max([abs(l) for c in clauses for l in c])
    var_count = [0] * (var_n + 1)
    max_c = -1
    for clause in clauses:
        max_c = max(len(clause), max_c)
        for lit in clause:
            var_count[abs(lit)] += 1

    if max_c <= 1:
        return clauses

    def remove_clauses(clause):
        if len(clause) == 1 and var_count[abs(clause[0])] <= 1:
            return False

        return True

    return list(filter(remove_clauses, clauses))


def build_dimacs_file(clauses: list, n_vars: int, comments: list = None):
    dimacs = []

    dimacs += comments if comments else []
    dimacs += [f"p cnf {n_vars} {len(clauses)}"]

    clauses = [elements_to_str(c) for c in clauses]
    dimacs += [f"{' '.join(c)} 0" for c in clauses]

    return "\n".join(dimacs)


def run_external_solver(input_dimacs: str, solver_exe: str = "binary/treengeling_linux") -> Tuple[bool, list]:
    """
    :param input_dimacs: Correctly formatted DIMACS file as string
    :param solver_exe: Absolute or relative path to solver executable [supports treengeling, lingeling, plingling]
    :return: returns True if formula is satisfiable and False otherwise, and solutions in form [1,2,-3, ...]
    """
    exe_path = Path(solver_exe).resolve()
    output = subprocess.run([str(exe_path)], input=input_dimacs, stdout=subprocess.PIPE, universal_newlines=True)
    satisfiable = [line for line in output.stdout.split("\n") if line.startswith("s ")]
    if len(satisfiable) > 1:
        raise ValueError("More than one satisifiability line returned!")

    is_sat = satisfiable[0].split()[-1]
    if is_sat != "SATISFIABLE" and is_sat != "UNSATISFIABLE":
        raise ValueError("Unexpected satisfiability value!")

    is_sat = is_sat == "SATISFIABLE"

    if is_sat:
        variables = [line[1:].strip() for line in output.stdout.split("\n") if line.startswith("v ")]
        solution = [int(var) for line in variables for var in line.split()][:-1]
    else:
        solution = []

    return is_sat, solution
