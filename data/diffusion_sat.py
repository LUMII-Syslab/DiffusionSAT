
from data.dimac import SatInstances, BatchedDimacsDataset
from data.SatSpecifics import SatSpecifics
from utils.DimacsFile import DimacsFile

from data.k_sat import KSatInstances

from satsolvers.Unigen import Unigen
from utils.AllSolutions import AllSolutions

from utils.DimacsFile import DimacsFile

def get_sat_solution(clauses: list):
    # by SK 2024: using the sat_solver_for_generators setting
    sat_solver = Unigen() # circular import: SatSolverRegistry().resolve(Config.sat_solver_for_generators)
    dimacs = DimacsFile(clauses=clauses)
    is_sat, solution = sat_solver.one_sample(str(dimacs))
    if not is_sat:
        raise ValueError("Can't get solution for UNSAT clauses")
    return solution


class DiffusionSatInstances(SatInstances):
    """ 
    """

    def __init__(self,                 
                 test_dimacs: DimacsFile,
                 solutions_multiplier_k=5, # the generator will generate (number of solutions * k) equal instances
                 train_and_validation_instances: SatInstances=None,                 
                 **kwargs) -> None:
        
        self.test_dimacs = test_dimacs
        self.solutions_multiplier_k = solutions_multiplier_k
        if train_and_validation_instances is None:
            train_and_validation_instances = KSatInstances(**kwargs)
        self.train_and_validation_instances = train_and_validation_instances

        self.p_k_2 = 0.3
        self.p_geo = 0.4

    def train_generator(self):
        for idx, (n_vars, clauses, *solution) in enumerate(self.train_and_validation_instances.train_generator()):
            solution = solution[0] if solution and solution[0] else get_sat_solution(clauses)
            yield n_vars, clauses, solution
        
        #return self.train_and_validation_instances.train_generator()
    
    def validation_generator(self):
        if hasattr(self.train_and_validation_instances, "validation_generator"):
            generator = self.train_and_validation_instances.validation_generator()
        else:
            generator = self.train_and_validation_instances.test_generator()
        for idx, (n_vars, clauses, *solution) in enumerate(generator):
            solution = solution[0] if solution and solution[0] else get_sat_solution(clauses)
            yield n_vars, clauses, solution

        #return self.train_and_validation_instances.validation_generator()

    def test_generator(self):
        all_solutions = AllSolutions(self.test_dimacs.number_of_vars(), self.test_dimacs.clauses())
        n_solutions = all_solutions.count()
        for i in range (n_solutions*self.solutions_multiplier_k):
            yield self.test_dimacs.number_of_vars(), self.test_dimacs.clauses()

   

class DiffusionSatDataset(BatchedDimacsDataset):
    def __init__(self, 
                 test_dimacs: DimacsFile,
                 solutions_multiplier_k=5, # the generator will generate (number of solutions * k) equal instances
                 train_and_validation_instances: SatInstances=None, 
                 min_vars=3, max_vars=30,
                 test_size=100, train_size=3000,
                 #test_size=10000, train_size=300000,
                 **kwargs):
        super().__init__(DiffusionSatInstances(test_dimacs, solutions_multiplier_k, train_and_validation_instances,
                                                min_vars=min_vars, max_vars=max_vars, test_size=test_size, train_size=train_size, **kwargs),
                         SatSpecifics(**kwargs), 
                         data_dir_suffix=str(min_vars)+"_"+str(max_vars))