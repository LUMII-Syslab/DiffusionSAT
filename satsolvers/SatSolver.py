from abc import abstractmethod, ABCMeta
from typing import Tuple

class SatSolver(metaclass=ABCMeta):
    """ Base class for internal/external SAT solvers.
    """

    @abstractmethod
    def one_sample(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the solution in the form [1,2,-3, ...], if the solution found or False and None, if not found
        """
        pass
        

    @abstractmethod
    def multiple_samples(self, input_dimacs: str, n_samples, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the list of solutions in the form [[1,2,-3, ...],...] for satisfiable SAT instances or False and [] for unsatisfiable
        """
        pass

    # to be used from subclasses:
    def _one_sample_from_multiple_samples(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        is_sat, samples = self.multiple_samples(input_dimacs, 1, **kwargs)
        if is_sat:
            return True, samples[-1]
        else:
            return False, None

    def _multiple_samples_from_one_sample(self, input_dimacs: str, n_samples, **kwargs) -> Tuple[bool, list]:
        if n_samples < 1:
            raise Exception("You should ask for at least one sample (n_samples>=1)")
        
        returned_samples = []
        for i in range(n_samples):
            is_sat, sample = self.one_sample(input_dimacs, **kwargs)
            if not is_sat:
                return False, []
            returned_samples.append(sample)
            
        return len(returned_samples)>0, returned_samples            
