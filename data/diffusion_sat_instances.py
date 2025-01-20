
from config import Config
from data.dimac import SatInstances, BatchedDimacsDataset
from data.SatSpecifics import SatSpecifics
from utils.DimacsFile import DimacsFile

from data.k_sat import KSatInstances

from satsolvers.Unigen import Unigen
from satsolvers.Glucose import Glucose

from utils.AllSolutions import AllSolutions

from utils.DimacsFile import DimacsFile
from utils.VariableAssignment import VariableAssignment

def get_sat_solution(n_vars, clauses: list):
    # by SK 2024:
    if Config.use_unigen:
        sat_solver = Unigen()
    else:
        sat_solver = Glucose() 
    # ^^^ Warning: there will be a circular import, if we write: SatSolverRegistry().resolve(Config.sat_solver_for_generators)

    dimacs = DimacsFile(n_vars=n_vars, clauses=clauses)
    is_sat, solution = sat_solver.one_sample(str(dimacs))
    if not is_sat:
        raise ValueError("Can't get solution for UNSAT clauses")
    if len(solution) != dimacs.number_of_vars():
        print("LENGTH MISMATCH:", len(solution), dimacs.number_of_vars())
        print(solution)
        raise Exception("Length mismatch")
    return solution


class DiffusionSatInstances(SatInstances):
    """ 
    """

    def __init__(self,                 
                 train_and_validation_instances: SatInstances=None,
                 test_dimacs: DimacsFile=None,
                 test_solutions_multiplier_k=5, # the generator will generate (AllSolutions(test_dimacs).count() * k) equal instances
                 **kwargs) -> None:
        
        self.test_dimacs = test_dimacs
        self.test_solutions_multiplier_k = test_solutions_multiplier_k
        if train_and_validation_instances is None:
            train_and_validation_instances = KSatInstances(**kwargs)
        self.train_and_validation_instances = train_and_validation_instances

        self.p_k_2 = 0.3
        self.p_geo = 0.4

    def train_generator(self):
        if self.train_and_validation_instances is None:
            raise Exception("This dataset does not include train instances.")

        for idx, (n_vars, clauses, *solution) in enumerate(self.train_and_validation_instances.train_generator()):
            solution = solution[0] if solution and solution[0] else get_sat_solution(n_vars, clauses)
            print("SOLUTION FOUND ",solution)
            yield n_vars, clauses, solution
        
        #return self.train_and_validation_instances.train_generator()
    
    def validation_generator(self):
        if self.train_and_validation_instances is None:
            raise Exception("This dataset does not include validation instances.")
        
        if hasattr(self.train_and_validation_instances, "validation_generator"):
            generator = self.train_and_validation_instances.validation_generator()
        else:
            generator = self.train_and_validation_instances.test_generator()
        for idx, (n_vars, clauses, *solution) in enumerate(generator):
            solution = solution[0] if solution and solution[0] else get_sat_solution(n_vars, clauses)            
            yield n_vars, clauses, solution

        #return self.train_and_validation_instances.validation_generator()

    def test_generator(self):
        if self.test_dimacs is None:
            raise Exception("This dataset does not include test instances.")

        all_solutions = AllSolutions(self.test_dimacs.number_of_vars(), self.test_dimacs.clauses())        
        n_solutions = all_solutions.count()

        #solution = get_sat_solution(self.test_dimacs.number_of_vars(), self.test_dimacs.clauses()) 
        #asgn = VariableAssignment(self.test_dimacs.number_of_vars(), self.test_dimacs.clauses())
        #asgn.assign_all_from_int_list(solution)

        for i in range (n_solutions*self.test_solutions_multiplier_k):
            if i % 1000 == 0:
                print("TEST #",i)
            yield self.test_dimacs.number_of_vars(), self.test_dimacs.clauses()#, solution

   

class DiffusionSatDataset(BatchedDimacsDataset):
    def __init__(self, 
                 train_and_validation_instances: SatInstances=None, 
                 test_dimacs: DimacsFile=None,
                 test_solutions_multiplier_k=5,
                 **kwargs):
        super().__init__(DiffusionSatInstances(train_and_validation_instances, test_dimacs, test_solutions_multiplier_k, **kwargs),
                         SatSpecifics(**kwargs), 
                         data_dir_suffix="all" if train_and_validation_instances is not None and test_dimacs is not None else
                                         "train" if train_and_validation_instances is not None else 
                                         "test" if test_dimacs is not None else
                                         "empty",
                         **kwargs)
        
        