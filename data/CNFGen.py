import random

import numpy as np
from cnfgen import RandomKCNF, CliqueFormula
import networkx as nx
from pysat.solvers import Cadical

from data.k_sat import KSATVariables


# todo: implement literals version
# todo: add more problems from cnfgen

class SAT_3(KSATVariables):
    """ Dataset with random 3-SAT instances at the satisfiability threshold from CNFGen library.
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


class Clique(KSATVariables):
    """ Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        super(KSATVariables, self).__init__(data_dir, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 10000
        self.test_size = 1000
        self.min_vertices = 4
        self.max_vertices = 20
        self.clique_size = 3

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        for _ in range(size):
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            # generate a random graph with such sparsity that a triangle is expected with probability 0.5.
            #eps = 0.2
            #p = 0.7 * ((1 + eps) * np.log(n_vertices)) / n_vertices # less exact formula
            p = 3 ** (1 / 3) / (n_vertices * (2 - 3 * n_vertices + n_vertices ** 2)) ** (1 / 3)

            it = 0
            while(True):
                it+=1
                G = nx.generators.erdos_renyi_graph(n_vertices, p)

                F = CliqueFormula(G, self.clique_size)
                n_vars = len(list(F.variables()))
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]
                with Cadical(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()

                if is_sat: break
            yield n_vars, iclauses
