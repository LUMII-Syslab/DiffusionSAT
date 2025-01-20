import random

import networkx as nx
import numpy as np
from cnfgen import RandomKCNF, CliqueFormula, DominatingSet, GraphColoringFormula
from pysat.solvers import Glucose4

from data.dimac import BatchedDimacsDataset, SatInstances
from data.SatSpecifics import SatSpecifics
from utils.sat import build_dimacs_file
from satsolvers.SatSolver import SatSolver
from satsolvers.Default import DefaultSatSolver

class SAT_3_Instances(SatInstances):
    """ Dataset with random 3-SAT instances at the satisfiability threshold from CNFGen library.
    """

    def __init__(self,
                 min_vars=5,
                 max_vars=30, # by SK #max_vars=100, 
                 # force_data_gen=False,
                 sat_solver: SatSolver=DefaultSatSolver(),
                 train_size = 100_000,
                 test_size = 5_000,
                 **kwargs) -> None:
        super().__init__()#min_vars=min_vars, max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        self.train_size = train_size
        self.test_size = test_size
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.sat_solver = sat_solver

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
                F._compress_clause = lambda x: x
                clauses = list(F.clauses())
                iclauses = [F._compress_clause(x) for x in clauses]

                dimacs = build_dimacs_file(iclauses, n_vars)
                
                is_sat, solution = self.sat_solver.one_sample(dimacs)

                if is_sat:
                    break

            yield len(solution), iclauses, solution

class SAT_3(BatchedDimacsDataset):
    def __init__(self, min_vars, max_vars, **kwargs):
        super().__init__(SAT_3_Instances(**kwargs), SatSpecifics(**kwargs))
        super().__init__(SAT_3_Instances(min_vars, max_vars, **kwargs), SatSpecifics(**kwargs), str(min_vars)+"_"+str(max_vars))
class Clique_Instances(SatInstances):
    """ Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=40, force_data_gen=False, **kwargs) -> None:
        super().__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                     force_data_gen=force_data_gen, **kwargs)
        self.train_size = 50000
        self.test_size = 10000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.clique_size_min = 3
        self.clique_size_max = 3

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

class Clique(BatchedDimacsDataset):
    def __init__(self, **kwargs):
        super().__init__(Clique_Instances(**kwargs), SatSpecifics(**kwargs))

class DomSet_Instances(SatInstances):
    """ Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=12, force_data_gen=False, **kwargs) -> None:
        super().__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
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

class DomSet(BatchedDimacsDataset):
    def __init__(self, **kwargs):
        super().__init__(DomSet_Instances(**kwargs), SatSpecifics(**kwargs))
class KColor_Instances(SatInstances):
    """
    Generates the clauses for colorability formula
    The formula encodes the fact that the graph :math:`G` has a coloring
    with color set ``colors``. This means that it is possible to
    assign one among the elements in ``colors``to that each vertex of
    the graph such that no two adjacent vertices get the same color.
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=20, force_data_gen=False, **kwargs) -> None:
        super().__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
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

class KColor(BatchedDimacsDataset):
    def __init__(self, **kwargs):
        super().__init__(KColor_Instances(**kwargs), SatSpecifics(**kwargs))
