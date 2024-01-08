# Run pytest from the project root

from AllSolutions import AllSolutions
from VariableAssignment import VariableAssignment

def test1():
    clauses = [[-1,2],[1,-2],[-3,4,5]]
    
    all = AllSolutions(5, clauses)
    print("approx count=", all.approximate_count())
    print("different solutions:")
    for i in all.as_int_list():
        asgn = VariableAssignment(5)
        asgn.assign_all_from_int(i)
        print(asgn.as_int_list())
    assert all.approximate_count() == 14
    
def test2():
    clauses = [[-1,2],[1,-2],[-3,4,5]]
    
    all = AllSolutions(5, clauses)
    print("count=", all.count())
    assert all.count() == 14


