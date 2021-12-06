from abc import abstractmethod

from data.CNFGen import SAT_3, Clique, DomSet, KColor, UNSAT_3
from data.PrimesGen import PrimesGen
from data.SHAGen import SHAGen
from data.SHAGen2019 import SHAGen2019
from data.k_sat import KSAT
from data.mixed_sat import MixGraphSAT
from data.sha_anf import ANF
from data.tsp import EuclideanTSP
from data.tsp_asymmetric import AsymmetricTSP
from model.neurocore import NeuroCore
from model.query_sat import QuerySAT
from model.query_sat_lit import QuerySATLit
from model.simple_neurosat import SimpleNeuroSAT
from model.tsp_matrix_se import TSPMatrixSE
from model.unsat_core_finder import CoreFinder


# from model.anf_sat import ANFSAT


class Registry:

    @property
    @abstractmethod
    def registry(self) -> dict:
        pass

    def resolve(self, name):
        if name in self.registry:
            return self.registry.get(name)

        raise ModuleNotFoundError(f"Class with name {name} is not registered in {self.__class__.__name__}!")

    @property
    def registered_names(self):
        return self.registry.keys()


class ModelRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "query_sat": QuerySAT,
            "query_sat_lit": QuerySATLit,
            "query_unsat": CoreFinder,
            "tsp_matrix_se": TSPMatrixSE,
            "simple_neuro_sat": SimpleNeuroSAT,
            # "anf_sat": ANFSAT,
            "neurocore": NeuroCore
        }


class SATDatasetRegistry(Registry):

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
            "mix_sat": MixGraphSAT,
            "sha-anf": ANF
        }


class UNSATDatasetRegistry(Registry):
    @property
    def registry(self) -> dict:
        return {
            "3-sat": UNSAT_3,
        }
