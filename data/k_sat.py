import random

import numpy as np
from pysat.solvers import Cadical

from data.dimac import DIMACDataset


class RandomKSAT(DIMACDataset):

    def __init__(self) -> None:
        self.dimacs_count = 20
        self.min_vars = 3
        self.max_vars = 10

        self.p_k_2 = 0.3
        self.p_geo = 0.4


    @staticmethod
    def __generate_k_iclause(n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

    # remove duplicate clauses
    # todo: remove subsumed clauses - when shorter clause is fully in a longer one, the longer one is redundant
    @staticmethod
    def prune(clauses):
        clauses_pruned = list({tuple(sorted(x)) for x in clauses})
        return clauses_pruned

    def dimac_generator(self):
        for _ in range(self.dimacs_count):
            n_vars = random.randint(self.min_vars, self.max_vars)

            solver = Cadical()
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

            iclause_unsat = iclause
            iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]

            iclauses.append(iclause_unsat)
            # yield n_vars, self.prune(iclauses) return only SAT instance

            iclauses[-1] = iclause_sat
            yield n_vars, self.prune(iclauses)
