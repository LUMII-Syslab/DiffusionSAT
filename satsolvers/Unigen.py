from typing import Tuple

from SatSolver import SatSolver
import subprocess
from pathlib import Path

class Unigen(SatSolver):

    def one_sample(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the solution in the form [1,2,-3, ...], if the solution found or False and None, if not found
        """
        return self._one_sample_from_multiple_samples(input_dimacs, **kwargs)

    def multiple_samples(self, input_dimacs: str, n_samples, unigen_exe: str = "binary/unigen_linux", seed=1, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the list of solutions in the form [[1,2,-3, ...],...] for satisfiable SAT instances or False and [] for unsatisfiable
        """
        exe_path = Path(unigen_exe).resolve()
        output = subprocess.run([str(exe_path), "--samples", str(n_samples), "--arjun", "0", "--seed", str(seed)], input=input_dimacs, stdout=subprocess.PIPE, universal_newlines=True)
        unsat_lines = [line for line in output.stdout.split("\n") if line.find("Formula was UNSAT")>=0]
        is_sat = len(unsat_lines) == 0
        sample_lines = [line for line in output.stdout.split("\n") if not line.startswith("c ") and not line.startswith("vp ") and not line==""]

        returned_samples = []
        if is_sat:
            if len(sample_lines)==0:
                raise ValueError("Unigen returned no solutions for a probably satisfiable instance")
            for chosen_line in sample_lines:
                sol = [int(var) for var in chosen_line.split()][:-1] # exclude the trailing zero
                returned_samples.append(sol)

        return is_sat, returned_samples
