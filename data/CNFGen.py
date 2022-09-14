import random

import networkx as nx
import numpy as np
from cnfgen import RandomKCNF, CliqueFormula, DominatingSet, GraphColoringFormula
from pysat.solvers import Glucose4

from data.k_sat import KSAT
from utils.sat import run_external_solver, run_unigen, build_dimacs_file

from config import Config


class SAT_3(KSAT):
    """ Dataset with random 3-SAT instances at the satisfiability threshold from CNFGen library.
    """

    def __init__(self, data_dir, min_vars=5, max_vars=100, force_data_gen=False, **kwargs) -> None:
        super(SAT_3, self).__init__(data_dir, min_vars=min_vars, max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 100000
        self.test_size = 5000
        self.min_vars = min_vars
        self.max_vars = max_vars

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            n_vars = random.randint(self.min_vars, self.max_vars)
            n_clauses = 4.258 * n_vars + 58.26 * np.power(n_vars, -2 / 3.)
            n_clauses = int(n_clauses)

            while True:
                F = RandomKCNF(3, n_vars, n_clauses)
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]

                dimacs = build_dimacs_file(iclauses, n_vars)

                if Config.use_unigen:
                    is_sat, solution = run_unigen(dimacs)
                else:
                    if n_vars > 200:
                        is_sat, solution = run_external_solver(dimacs)
                    else:
                        with Glucose4(bootstrap_with=iclauses) as solver:
                            is_sat = solver.solve()
                            solution = None

                if is_sat:
                    break

            yield n_vars, iclauses, solution


class Clique(KSAT):
    """ Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=20, force_data_gen=False, **kwargs) -> None:
        super(Clique, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                     force_data_gen=force_data_gen, **kwargs)
        self.train_size = 50000
        self.test_size = 10000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.clique_size_min = 3
        self.clique_size_max = 6

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            # generate a random graph with such sparsity that a triangle is expected with probability 0.5.
            # eps = 0.2
            # p = 0.7 * ((1 + eps) * np.log(n_vertices)) / n_vertices # less exact formula
            p = 3 ** (1 / 3) / (n_vertices * (2 - 3 * n_vertices + n_vertices ** 2)) ** (1 / 3)

            it = 0
            while True:
                it += 1
                G = nx.generators.erdos_renyi_graph(n_vertices, p)

                F = CliqueFormula(G, random.randint(self.clique_size_min, self.clique_size_max))
                n_vars = len(list(F.variables()))
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]
                with Glucose4(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()

                if is_sat:
                    break
            yield n_vars, iclauses


class DomSet(KSAT):
    """ Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=12, force_data_gen=False, **kwargs) -> None:
        super(DomSet, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                     force_data_gen=force_data_gen, **kwargs)
        self.train_size = 10000
        self.test_size = 5000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            p = 0.2
            domset_size = (n_vertices + 2) // 3

            it = 0
            while True:
                it += 1
                G = nx.generators.erdos_renyi_graph(n_vertices, p)

                F = DominatingSet(G, domset_size)
                n_vars = len(list(F.variables()))
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]
                with Glucose4(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()

                if is_sat:
                    break
            # print(n_vertices, n_vars, len(iclauses), it)

            yield n_vars, iclauses


class KColor(KSAT):
    """
    Generates the clauses for colorability formula
    The formula encodes the fact that the graph :math:`G` has a coloring
    with color set ``colors``. This means that it is possible to
    assign one among the elements in ``colors``to that each vertex of
    the graph such that no two adjacent vertices get the same color.
    """

    def __init__(self, data_dir, min_vertices=5, max_vertices=20, force_data_gen=False, **kwargs) -> None:
        super(KColor, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                     force_data_gen=force_data_gen, **kwargs)
        self.train_size = 50000
        self.test_size = 10000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            p = 0.5
            n_colors = (n_vertices // 5) + 1  # Approximate formula to describe the relation
            if n_colors == 2:
                n_colors = 3

            it = 0
            while True:
                it += 1
                G = nx.generators.erdos_renyi_graph(n_vertices, p)

                F = GraphColoringFormula(G, n_colors)
                n_vars = len(list(F.variables()))
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]
                with Glucose4(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()

                if is_sat:
                    break

            yield n_vars, iclauses
