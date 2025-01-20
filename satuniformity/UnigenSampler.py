import tensorflow as tf
from utils.DimacsFile import DimacsFile
from satsolvers.Unigen import Unigen
from utils.VariableAssignment import VariableAssignment

class UnigenSampler():

    def __init__(self, dimacs_file: DimacsFile):
        self.dimacs_file = dimacs_file


    def samples(self, n_samples): 
        """
        :param n_samples: how many correct samples to generate
        :return: returns the dict: solution as int => count
        """

        (is_sat, unigen_samples) = Unigen().multiple_samples(
            str(self.dimacs_file), n_samples=n_samples
        )

        # random.shuffle(unigen_samples) - subsampling; better not to use
        unigen_samples = unigen_samples[0:n_samples]  # getting exactly n_samples solutions
        asgn = VariableAssignment(clauses=self.dimacs_file.clauses())

        n_solutions = 0
        unigen_dict = {}
        for sample in unigen_samples:
            asgn.assign_all_from_int_list(sample)
            i_sample = int(asgn)
            if not i_sample in unigen_dict:
                unigen_dict[i_sample] = 0
                n_solutions += 1
            unigen_dict[i_sample] += 1
        return unigen_dict
    