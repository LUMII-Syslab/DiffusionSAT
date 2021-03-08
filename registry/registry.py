from abc import abstractmethod

from data.CNFGen import KColor, MixGraphSAT
from data.CNFGen import SAT_3, Clique, DomSet
from data.PrimesGen import PrimesGen
from data.SHAGen import SHAGen
from data.SHAGen2019 import SHAGen2019
from data.k_sat import KSAT
from data.tsp import EuclideanTSP
from data.tsp_asymmetric import AsymmetricTSP
from model.attention_sat import AttentionSAT
from model.neuro_sat import NeuroSAT
from model.query_sat import QuerySAT
from model.query_sat_lit import QuerySATLit
from model.simple_neurosat import SimpleNeuroSAT
from model.tsp_matrix_se import TSPMatrixSE


class Registry:

    @property
    @abstractmethod
    def registry(self) -> dict:
        pass

    def resolve(self, name):
        if name in self.registry:
            return self.registry.get(name)

        raise ModuleNotFoundError(f"Model with name {name} is not registered!")

    @property
    def registered_names(self):
        return self.registry.keys()


class ModelRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "attention_sat": AttentionSAT,
            "query_sat": QuerySAT,
            "query_sat_lit": QuerySATLit,
            "neuro_sat": NeuroSAT,
            "tsp_matrix_se": TSPMatrixSE,
            "simple_neuro_sat": SimpleNeuroSAT
        }


class DatasetRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "k_sat": KSAT,
            "k_color": KColor,
            "3-sat": SAT_3,
            "clique": Clique,
            "dominating_set": DomSet,
            "sha-gen": SHAGen,
            "sha-gen2019": SHAGen2019,
            "euclidean_tsp": EuclideanTSP,
            "asymmetric_tsp": AsymmetricTSP,
            "primes": PrimesGen,
            "mix_sat": MixGraphSAT
        }
