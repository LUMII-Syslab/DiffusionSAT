from typing import Tuple

from SatSolver import SatSolver
from utils.sat import run_external_solver

class Treengeling(SatSolver):

    def one_sample(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the solution in the form [1,2,-3, ...], if the solution found or False and None, if not found
        """
        is_sat, solution = run_external_solver(input_dimacs)
        if not is_sat:
            solution = None
        return is_sat, solution

    def multiple_samples(self, input_dimacs: str, n_samples, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the list of solutions in the form [[1,2,-3, ...],...] for satisfiable SAT instances or False and [] for unsatisfiable
        """
        return self._multiple_samples_from_one_sample(input_dimacs, n_samples, **kwargs)
    