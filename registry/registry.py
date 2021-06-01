from abc import abstractmethod

from data.CNFGen import SAT_3, Clique, DomSet, KColor
from data.SHAGen2019 import SHAGen2019
from data.k_sat import KSAT
from model.neurocore import NeuroCore
from model.query_sat import QuerySAT
from model.simple_neurosat import SimpleNeuroSAT


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
            "simple_neuro_sat": SimpleNeuroSAT,
            "neurocore": NeuroCore
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
            "sha-gen2019": SHAGen2019,
        }
