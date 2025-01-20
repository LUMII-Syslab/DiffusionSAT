import tensorflow as tf
from utils.DimacsFile import DimacsFile
from utils.VariableAssignment import VariableAssignment
from satsolvers.QuickSampler import QuickSampler as QuickSolver

class QuickSampler():

    def __init__(self, dimacs_file: DimacsFile):
        self.dimacs_file = dimacs_file


    def samples(self, n_samples): 
        """
        :param n_samples: how many correct samples to generate
        :return: returns the dict: solution as int => count
        """

        (is_sat, quicksampler_samples) = QuickSolver().multiple_samples(
            str(DimacsFile(clauses=self.dimacs_file.clauses())), n_samples=n_samples
        )

        quicksampler_dict = {}
        asgn = VariableAssignment(clauses=self.dimacs_file.clauses())
        for sample in quicksampler_samples:
            asgn.assign_all_from_int_list(sample)
            i_sample = int(asgn)
            if not i_sample in quicksampler_dict:
                quicksampler_dict[i_sample] = 0
            quicksampler_dict[i_sample] += 1

        return quicksampler_dict