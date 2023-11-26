#!/usr/bin/env python3

import numpy as np


class VariableAssignment:
    def __init__(self, n_vars=0, clauses=[]):
        if type(clauses) != list:
            clauses = self.__raggedtensor_to_list(clauses)
        if n_vars == 0:
            n_vars = self.__number_of_vars(clauses)
        self.x = [False] * n_vars
        self.clauses = clauses

    def __raggedtensor_to_list(self, tf_clauses):
        n_clauses = np.shape(tf_clauses)[0]
        clauses = []
        for i in range(n_clauses):
            tf_clause = tf_clauses[i]
            tf_clause_len = len(tf_clause)  # tf.shape(tf_clause)[0]

            clause = []
            for j in range(tf_clause_len):
                clause.append(int(tf_clause[j]))

            clauses.append(clause)
        return clauses

    def __number_of_vars(self, clauses):
        a = np.array(self.__flatten(clauses))
        return max(abs(a.max()), abs(a.min()))

    def __flatten(self, l):
        return [item for sublist in l for item in sublist]

    def assign(self, i, value):
        # i must be zero-based
        self.x[i] = value

    def assign_all(self, x):
        self.x = x

    def assign_all_from_int_list(self, x):
        for i in x:
            self.assign(abs(i) - 1, i > 0)

    def assign_all_from_bit_list(self, x):
        bit_no = 0
        for bit in x:
            i_bit = int(bit)
            self.assign(bit_no, i_bit == 1)
            bit_no += 1

    def __int__(self):
        # right-to-left binary encoding
        res = 0
        for bit_no in range(len(self.x)):
            if self.x[bit_no]:
                res = res | (1 << bit_no)
        return res

    def __str__(self):
        return str(int(self))

    def satisfiable(self):
        for a in range(len(self.clauses)):  # a = clause index
            c = self.clauses[a]
            is_clause_satisfied = False
            for literal in c:
                i = abs(literal) - 1  # zero-based
                if (literal > 0) == self.x[i]:
                    is_clause_satisfied = True
                    break  # the clause is satisfied by x[i], no need to check other vars
            if not is_clause_satisfied:
                return False  # clause c cannot be satisfied
        return True
    
    def as_int_list(self):
        result = []
        for i in range(len(self.x)):
            if self.x[i]:
                result.append(i+1)
            else:
                result.append(-(i+1))
        return result


if __name__ == "__main__":  # test
    a = VariableAssignment(3, [])
    a.assign_all_from_int_list([1, 2, 3])
    print(int(a))
