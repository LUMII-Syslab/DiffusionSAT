#!/usr/bin/env python3

import os
import fnmatch
from natsort import natsorted, ns

from data.dimac import SatInstances, BatchedDimacsDataset
from data.SatSpecifics import SatSpecifics

from utils.DimacsFile import DimacsFile

# from utils.sat import remove_unused_vars
MY_DIR = os.path.dirname(os.path.realpath(__file__))

class SatLib_Instances(SatInstances):
    """
    Enumerating SAT instances obtained from https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html.

    """

    def __init__(self, data_dir,
                 test_size_factor=0.10, # 10% of data for tests
                 max_nodes_per_batch=5000,
                 **kwargs) -> None:
#        super(SplotData, self).__init__(data_dir, force_data_gen=False, **kwargs)
        super().__init__(data_dir, input_mode='literals', max_nodes_per_batch=0)

        self.data_dir = os.path.join(MY_DIR,"satlib")#data_dir
        print(f"We are in satlib data init; data_dir={self.data_dir}")
        self.test_size_factor = test_size_factor


    def train_generator(self) -> tuple:
        return self.__generator()

    def test_generator(self) -> tuple:
        return self.__generator(True)
    

    def __generator(self, is_test=False) -> tuple:        
        files = natsorted( {x for x in fnmatch.filter(os.listdir(self.data_dir), '*.cnf')}, alg=ns.IGNORECASE )

        k = int(1/self.test_size_factor) # each k-th file will be a part of the test set; others form the train set

        i = 0
        for fname in files:
            i += 1
            if is_test == (i%k == 0):  # is_test <=> we are a k-th file

                dimacs = DimacsFile(os.path.join(self.data_dir,fname))
                dimacs.load()
                yield dimacs.number_of_vars(), dimacs.clauses()


class SatLib(BatchedDimacsDataset):

    def __init__(self, **kwargs):
        super().__init__(SatLib_Instances(**kwargs), SatSpecifics(**kwargs))


if __name__ == "__main__":

    

    # for 
    d = SatLib(os.path.join(MY_DIR,"qqq"), 0.1)
    d.__generator(d, True)

    #nvars, clauses = remove_unused_vars(10, [[-1, 4, 9], [-4, 1, -10], [10]])
    # nvars, clauses = remove_unused_vars(3, [[-1,3]])
    # nvars, clauses = remove_unused_vars(3, [[]])
    #print(nvars, clauses)
