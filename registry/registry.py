from abc import abstractmethod

from data.CNFGen import SAT_3, Clique, DomSet, KColor
from data.PrimesGen import PrimesGen
from data.SHAGen import SHAGen
from data.SHAGen2019 import SHAGen2019
from data.k_sat import KSAT
from data.mixed_sat import MixGraphSAT
from data.splot import SplotData
from data.satlib import SatLib
#from data.tsp import EuclideanTSP
#from data.tsp_asymmetric import AsymmetricTSP
from data.sha_anf import ANF
from model.attention_sat import AttentionSAT
from model.neuro_sat import NeuroSAT
from model.neurocore import NeuroCore
from model.query_sat import QuerySAT
from model.query_sat_lit import QuerySATLit
from model.simple_neurosat import SimpleNeuroSAT
#from model.tsp_matrix_se import TSPMatrixSE
from model.anf_sat import ANFSAT

from satsolvers.QuickSampler import QuickSampler
from satsolvers.Unigen import Unigen
from satsolvers.Walksat import Walksat
from satsolvers.Treengeling import Treengeling
from satsolvers.Lingeling import Lingeling
from satsolvers.Default import DefaultSatSolver
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
            # "tsp_matrix_se": TSPMatrixSE,
            "simple_neuro_sat": SimpleNeuroSAT,
            "anf_sat":ANFSAT,
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
            "sha-gen": SHAGen,
            "sha-gen2019": SHAGen2019,
            # "euclidean_tsp": EuclideanTSP,
            # "asymmetric_tsp": AsymmetricTSP,
            "primes": PrimesGen,
            "mix_sat": MixGraphSAT,
            "sha-anf": ANF,
            "splot": SplotData,
            "satlib": SatLib
        }

class SatSolverRegistry(Registry):
    @property
    def registry(self) -> dict:
        return {
            "quicksampler": QuickSampler,
            "unigen": Unigen,
            "walksat": Walksat,
            "treengeling": Treengeling,
            "lingeling": Lingeling,
            "default": DefaultSatSolver
        }
    
    