from abc import abstractmethod

from data.sat import NeuroSATDataset
from model.feedforward_sat import FeedForwardSAT
from model.neuro_sat import NeuroSAT
from model.query_sat import QuerySAT


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
            "feedforward_sat": FeedForwardSAT
        }


class DatasetRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {"random_sat": NeuroSATDataset}
