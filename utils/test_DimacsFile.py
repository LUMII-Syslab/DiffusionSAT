# Run pytest from the project root

from DimacsFile import DimacsFile

def test_dimacs1():
    df = DimacsFile(clauses=[[1,2],[1,2],[3]])
    df.reduce_clauses()
    
    assert df.clauses()==[[3],[1,2]] # sorted by len


def test_dimacs2():
    df = DimacsFile(clauses=[[1,2,3],[1,2],[3]])
    df.reduce_clauses()
    
    assert df.clauses()==[[3],[1,2]] # sorted by len
    
    
def test_dimacs3():
    df = DimacsFile(clauses=[[1,2,-3],[1,-2],[1]])
    df.reduce_clauses()
    
    assert df.clauses()==[[1]] # the result will be sorted