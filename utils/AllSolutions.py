from utils.DimacsFile import DimacsFile
from utils.VariableAssignment import VariableAssignment
from pyunigen import Sampler    # for counting SAT solutions
from pyapproxmc import Counter  # for counting SAT solutions

from satsolvers.Unigen import Unigen

class AllSolutions:
    """ The class representing all SAT solutions.
        Uses Unigen internally.
    """

    def __init__(self, n_vars: int, clauses: list):
        self.n_vars = n_vars
        self.clauses = clauses
        self.__approximate_count = -1
        self.__count = -1
        self.__solutions_as_ints = set()

    def approximate_count(self):
        """
        :return: returns the approximate number of solutions
        """
        if self.__approximate_count >= 0:
            return self.__approximate_count

        # via pyunigen
        c = Sampler()
        for clause in self.clauses:
            c.add_clause(clause)
        cells, hashes, samples = c.sample(num=0)
        result1 = cells * 2**hashes
    
        # via Counter
        counter = Counter(seed=2157, epsilon=0.5, delta=0.15)
        for clause in self.clauses:
            counter.add_clause(clause)
        cell_count, hash_count = counter.count()        
        result2 = cell_count * (2**hash_count)

        return max(result1, result2)


    def count(self):
        if self.__count >= 0:
            return self.__count

        k = 5  # for statistical significance, we need at least k=5
        gen_cnt = self.approximate_count() * k
        is_sat, unigen_samples = Unigen().multiple_samples(
            str(DimacsFile(clauses=self.clauses)), n_samples=gen_cnt
        )

        if not is_sat:
            self.__count = 0
            return self.__count
        
        self.__count = 0 # will increment below

        for sample in unigen_samples:
            asgn = VariableAssignment(clauses=self.clauses)
            asgn.assign_all_from_int_list(sample)
            i_sample = int(asgn)
            if not i_sample in self.__solutions_as_ints:
                self.__solutions_as_ints.add(i_sample)
                self.__count += 1        

        return self.__count

    def as_int_list(self):
        # the list of variable assignments in the right-to-left binary encoding
        self.count() # populates __solutions_as_ints
        return list(self.__solutions_as_ints)
        

