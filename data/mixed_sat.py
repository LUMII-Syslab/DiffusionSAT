import random

from data.CNFGen import Clique, DomSet, KColor, SAT_3
from data.SHAGen2019 import KSAT, SHAGen2019


class MixGraphSAT(KSAT):
    def __init__(self, data_dir, min_vertices=5, max_vertices=20, force_data_gen=False, **kwargs) -> None:
        super(MixGraphSAT, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                          force_data_gen=force_data_gen, **kwargs)
        self.train_size = 30000
        self.test_size = 5000
        self.datasets = [Clique(data_dir, min_vertices, max_vertices),
                         DomSet(data_dir, min_vertices, max_vertices),
                         KColor(data_dir, min_vertices, max_vertices),
                         KSAT(data_dir, min_vars=3, max_vars=100),
                         SAT_3(data_dir, min_vars=4, max_vars=100),
                         SHAGen2019(data_dir)]

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            dataset = random.choice(self.datasets)  # type: KSAT
            yield from dataset._generator(1)
