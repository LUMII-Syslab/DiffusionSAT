import random
from itertools import islice

from data.CNFGen import Clique, DomSet, KColor, SAT_3

from data.dimac import SatInstances, BatchedDimacsDataset
from data.SatSpecifics import SatSpecifics

class MixGraphSAT_Instances(SatInstances):
    def __init__(self, data_dir, min_vertices=5, max_vertices=20, force_data_gen=False, **kwargs) -> None:
        super().__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                          force_data_gen=force_data_gen, **kwargs)
        self.train_size = 60000
        self.test_size = 10000
        self.datasets = [Clique(data_dir),
                         DomSet(data_dir, min_vertices=4, max_vertices=20),
                         KColor(data_dir),
                         KSatDataset(data_dir, min_vars=3, max_vars=100),
                         SAT_3(data_dir, min_vars=5, max_vars=100)]

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            dataset = random.choice(self.datasets)  # type: KSAT
            yield from islice(dataset._generator(1), 1)

class MixGraphSAT(BatchedDimacsDataset):
    def __init__(self, **kwargs):
        super().__init__(MixGraphSAT_Instances(**kwargs), SatSpecifics(**kwargs))