from utils.DimacsFile import DimacsFile
from utils.VariableAssignment import VariableAssignment
import numpy as np
import random
from satsolvers.Default import DefaultSatSolver

def shuffle(source: DimacsFile, target: DimacsFile):
    n = source.number_of_vars()
    perm = [x+1 for x in range(n)]
    random.shuffle(perm)
    perm = [0]+perm # adding dummy item perm[0]=0

    map = dict()    

    for i in range(1,n+1):
        map[i] = perm[i]
        map[-i] = -perm[i]

    for old_clause in source.clauses():
        new_clause = [map[x] for x in old_clause]
        target.add_clause(new_clause)

    target.add_comment("permutation was "+str(perm[1:]))
    solver = DefaultSatSolver()
    ok, sol = solver.one_sample(str(target))
    if ok:
        asgn = VariableAssignment(n)
        asgn.assign_all_from_int_list(sol)
        bit_str = str(asgn)
        target.add_comment("sol "+' '.join(bit_str))


source = DimacsFile(filename="test3.dimacs")
source.load()
target = DimacsFile(filename="test3f.dimacs", n_vars=source.number_of_vars())
shuffle(source, target)
target.store()
