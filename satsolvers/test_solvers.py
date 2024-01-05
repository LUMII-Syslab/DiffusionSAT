# Run pytest from the project root

from utils.sat import build_dimacs_file
from QuickSampler import QuickSampler
from Unigen import Unigen
from Walksat import Walksat
from Treengeling import Treengeling
from Lingeling import Lingeling
from Default import DefaultSatSolver

def test_quicksampler1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = QuickSampler().one_sample(dimacs)
    print("quicksampler result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]

def test_unigen1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = Unigen().one_sample(dimacs)
    print("unigensampler result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]
    
def test_unigen2():
    dimacs = build_dimacs_file([[-1],[1]], 1)
    print(dimacs)
    
    is_sat, solution = Unigen().one_sample(dimacs)
    print("unigensampler result:")
    print(is_sat, solution)
    assert not is_sat and solution is None

def test_walksat1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = Walksat().one_sample(dimacs)
    print("walksat result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]

def test_treengeling1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = Treengeling().one_sample(dimacs)
    print("Treengeling result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]

def test_lingeling1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = Lingeling().one_sample(dimacs)
    print("Lingeling result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]

def test_default1():
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)
    print(dimacs)
    
    is_sat, solution = DefaultSatSolver().one_sample(dimacs)
    print("DefaultSatSolver result:")
    print(is_sat, solution)
    assert solution == [-1,-2] or solution == [1,2]
