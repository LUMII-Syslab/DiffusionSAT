import random

import numpy as np
from cnfgen import RandomKCNF
from pysat.solvers import Cadical

from data.k_sat import KSATVariables


# todo: implement literals version
# todo: add more problems from cnfgen

class SAT_3(KSATVariables):
    """ Dataset with random 3-SAT (hard?) instances from CNFGen library.
    """

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        super(KSATVariables, self).__init__(data_dir, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 100000
        self.test_size = 5000
        self.min_vars = 5
        self.max_vars = 40

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        for _ in range(size):
            n_vars = random.randint(self.min_vars, self.max_vars)
            n_clauses = 4.258 * n_vars + 58.26 * np.power(n_vars, -2 / 3.)

            while True:
                F = RandomKCNF(3, n_vars, n_clauses)
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]
                with Cadical(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()

                if is_sat: break

            yield n_vars, iclauses
