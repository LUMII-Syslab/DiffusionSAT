import random

import numpy as np
from pysat.solvers import Solver # using Solver since Cadical is not present in recent pysat versions

from data.dimac import SatInstances, BatchedDimacsDataset
from data.SatSpecifics import SatSpecifics
from utils.DimacsFile import DimacsFile
import math
import random

class KSatInstances(SatInstances):
    """ Dataset from NeuroSAT paper, just for variables. Dataset generates k-SAT
    instances with variable count in [min_size, max_size].
    """

    def __init__(self,
                 min_vars=3, max_vars=30,
                 test_size=10000, train_size=300000,
                 desired_multiplier_for_the_number_of_solutions=10, **kwargs) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.desired_multiplier_for_the_number_of_solutions=desired_multiplier_for_the_number_of_solutions

        self.p_k_2 = 0.3
        self.p_geo = 0.4

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size): # -> tuple:

        for j in range(size):
            n_vars = random.randint(self.min_vars, self.max_vars)

            solver = Solver() # Solver() instead of Cadical() by SK
            iclauses = []

            while True:
                k_base = 1 if random.random() < self.p_k_2 else 2
                k = k_base + np.random.geometric(self.p_geo)
                iclause = self.__generate_k_iclause(n_vars, k)

                solver.add_clause(iclause)
                is_sat = solver.solve()

                if is_sat:
                    iclauses.append(iclause)
                else:
                    break

            # Since { c[1], . . . , c[m−1]} had a satisfying assignment, negating a single literal in c[m] must yield a satisﬁable problem { c[1], . . . , c[m−1], c[m]′} 
            # // from the NeuroSAT paper
            iclause_unsat = iclause
            iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]
            

            iclauses.append(iclause_unsat)
            # yield only SAT instance
            # yield n_vars, self.prune(iclauses)

            iclauses[-1] = iclause_sat
            iclauses = self.remove_duplicate_clauses(iclauses)

            if self.desired_multiplier_for_the_number_of_solutions > 1:
                # REMOVING SOME CLAUSES TO INCREASE THE NUMBER OF SOLUTIONS:
                m = len(iclauses)
                # m clauses lead to ~1 solution
                # 0 clauses lead to 2^n solutions
                # Thus, each new clause divides the number of solutions by ~ x=2^(n/m); each removed clause multiplies the number of solutions by x.
                # We want self.desired_multiplier_for_the_number_of_solutions. Thus, we remove log_x(self.desired_multiplier_for_the_number_of_solutions) clauses.
                x = pow(2, n_vars*1.0/m)
                d = round(math.log(self.desired_multiplier_for_the_number_of_solutions, x), 0)
                d = min(d, m-1) # remove at most m-1 clauses (keep at least 1 clause)
                d = max(d, 0) # remove at least 0 clauses (avoid negative d)
                d = int(d)

                indices_to_remove = random.sample(range(m), d)
                indices_to_remove = sorted(indices_to_remove, reverse=True)
                for i in indices_to_remove:
                    iclauses = iclauses[:i] + iclauses[i+1:] # remove the i-th clause

            yield n_vars, iclauses

    @staticmethod
    def __generate_k_iclause(n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

    
    @staticmethod
    def remove_duplicate_clauses(clauses):
        df = DimacsFile(clauses=clauses)
        df.reduce_clauses() # also removes subsumed clauses
        return df.clauses()
        

class KSatDataset(BatchedDimacsDataset):
    def __init__(self, min_vars, max_vars, **kwargs):
        super().__init__(KSatInstances(min_vars, max_vars, **kwargs), SatSpecifics(**kwargs), str(min_vars)+"_"+str(max_vars))