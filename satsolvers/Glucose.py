from typing import Tuple

from satsolvers.SatSolver import SatSolver
from pysat.solvers import Glucose4
from utils.DimacsFile import DimacsFile

class Glucose(SatSolver):

    def one_sample(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the solution in the form [1,2,-3, ...], if the solution found or False and None, if not found
        """
        df = DimacsFile()
        df.load_from_string(input_dimacs)
        with Glucose4(bootstrap_with=df.clauses()) as solver:
            is_sat = solver.solve()
            if not is_sat:
                return False, None
            
            solution = solver.get_model()
            n_vars = df.number_of_vars()
            while len(solution)<n_vars:                
                solution.append(len(solution)+1) # append the False value for missing variables

            return True, solution

    def multiple_samples(self, input_dimacs: str, n_samples, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the list of solutions in the form [[1,2,-3, ...],...] for satisfiable SAT instances or False and [] for unsatisfiable
        """
        return self._multiple_samples_from_one_sample(input_dimacs, n_samples, **kwargs)
    