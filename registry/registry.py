from abc import abstractmethod

from data.k_sat import KSATVariables, KSATLiterals
from data.tsp import EuclideanTSP
from model.matrix_se import MatrixSE
from model.neuro_sat import NeuroSAT
from model.query_sat import QuerySAT
from data.CNFGen import SAT_3


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
            "query_sat": QuerySAT,
            "neuro_sat": NeuroSAT,
            "matrix_se": MatrixSE
        }


class DatasetRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "k_sat_variables": KSATVariables,
            "k_sat_literals": KSATLiterals,
            "3-sat":SAT_3,
            "euclidean_tsp": EuclideanTSP
        }
