#!/usr/bin/env python3

import subprocess
import os
import uuid
from typing import Tuple
from pathlib import Path
from utils.VariableAssignment import VariableAssignment
from utils.DimacsFile import DimacsFile

from utils.sat import run_external_solver, run_unigen, build_dimacs_file

def run_quicksampler(input_dimacs: str, solver_exe: str = "binary/quicksampler_linux", seed=1, n_samples=1) -> Tuple[bool, list]: # added by SK@2023-11
    """
    :param input_dimacs: Correctly formatted DIMACS file as string
    :param solver_exe: Absolute or relative path to solver executable
    :return: returns True if formula is satisfiable and False otherwise, and an almost uniformly generated solution in form [1,2,-3, ...]
    """
    exe_path = Path(solver_exe).resolve()
    dimacs_path = str(uuid.uuid4())+".dimacs"
    samples_path = dimacs_path+".samples"

    with open(dimacs_path, 'w') as file:
        file.write(input_dimacs)

    df = DimacsFile(filename=dimacs_path)
    df.load()

    remaining = n_samples
    solution = []

    while remaining > 0:
        subprocess.run([str(exe_path), "-n", str(n_samples), dimacs_path])

        if not os.path.exists(samples_path):
            raise Exception("Quicksampler returned no result")
        
        # reading the file line by line:
        with open(samples_path, 'r') as file:
            for line in file:
                arr = line.split()
                if len(arr)>=2:
                    bit_str = arr[1] # the second line element is bit-encoded solution
                    print(bit_str) 
                    bit_list = [int(char) for char in bit_str]
                    asgn = VariableAssignment(len(bit_str), df.clauses())
                    asgn.assign_all_from_bit_list(bit_list)
                    if asgn.satisfiable():
                        # append only satisfiable solution
                        remaining -= 1
                        sol = asgn.as_int_list()
                        if n_samples==1:
                            solution = sol
                        else:
                            solution.append(sol)
        
        try:
            os.remove(samples_path)
        except Exception as e:
            pass

    

    try:
        os.remove(dimacs_path)
    except Exception as e:
        pass

    return len(solution)>0, solution


if __name__ == '__main__':
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)

#    is_sat, solution = run_pyunigen(dimacs)
    is_sat, solution = run_quicksampler(dimacs, n_samples=1)
    print("quicksampler")
    print(is_sat, solution)


    is_sat, solution = run_unigen(dimacs)
    print("unigen")
    print(is_sat, solution)

    print("lingeling")
    is_sat, solution = run_external_solver(dimacs)
    print(is_sat, solution)
