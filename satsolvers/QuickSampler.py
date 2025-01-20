from typing import Tuple

import subprocess
from pathlib import Path
import os
import uuid

import random
from satsolvers.SatSolver import SatSolver
from utils.VariableAssignment import VariableAssignment
from utils.DimacsFile import DimacsFile


class QuickSampler(SatSolver):

    def one_sample(self, input_dimacs: str, **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the solution in the form [1,2,-3, ...], if the solution found or False and None, if not found
        """
        return self._one_sample_from_multiple_samples(input_dimacs, **kwargs)

    def multiple_samples(self, input_dimacs: str, n_samples, solver_exe: str = "binary/quicksampler_linux", **kwargs) -> Tuple[bool, list]:
        """
        :param input_dimacs: Correctly formatted DIMACS file as string (including \\n)
        :return: returns True and the list of solutions in the form [[1,2,-3, ...],...] for satisfiable SAT instances or False and [] for unsatisfiable
        """
        exe_path = Path(solver_exe).resolve()
        dimacs_path = str(uuid.uuid4())+".dimacs"
        samples_path = dimacs_path+".samples"

        with open(dimacs_path, 'w') as file:
            file.write(input_dimacs)

        df = DimacsFile(filename=dimacs_path)
        df.load()

        remaining = n_samples
        returned_samples = []

        while remaining > 0:
            subprocess.run([str(exe_path), "-n", str(n_samples), dimacs_path])

            if not os.path.exists(samples_path):
                raise Exception("Quicksampler returned no result")
            
            i=1
            # reading the file line by line:
            with open(samples_path, 'r') as file:
                lines = file.readlines()
                random.shuffle(lines)
                #for line in file: - without shuffle
                for line in lines:
                    arr = line.split()
                    if len(arr)>=2:
                        bit_str = arr[1] # the second line element is bit-encoded solution
                        bit_list = [int(char) for char in bit_str]
                        asgn = VariableAssignment(len(bit_str), df.clauses())
                        asgn.assign_all_from_bit_list(bit_list)
                        i+=1
                        if asgn.satisfiable():
                            # append only satisfiable solution
                            sol = asgn.as_int_list()
                            returned_samples.append(sol)
                            remaining -= 1
                            if remaining == 0:
                                break
            
            try:
                os.remove(samples_path)
            except Exception as e:
                pass

        

        try:
            os.remove(dimacs_path)
        except Exception as e:
            pass

        return len(returned_samples)>0, returned_samples

