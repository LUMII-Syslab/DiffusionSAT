#!/usr/bin/env python3

import numpy as np


class DimacsFile:
    def __init__(self, filename="file.cnf", n_vars=0, clauses=[]):
        if type(clauses) != list:
            clauses = self.__raggedtensor_to_list(clauses)
        if n_vars == 0:
            if (len(clauses)>0):
                n_vars = self.__number_of_vars(clauses)
        self.filename = filename
        self.n_vars = n_vars
        self.i_clauses = clauses
        self.b_values = {}

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
        return int(max(abs(a.max()), abs(a.min())))

    def __flatten(self, l):
        return [item for sublist in l for item in sublist]

    def load(self):
        f = open(self.filename, "r")
        lines = f.readlines()
        f.close()

        self.n_vars = 0
        self.i_clauses = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            i = line.find("p cnf")
            if i >= 0:
                line = line[len("p cnf") :].strip()
                i = line.find(" ")
                line = line[:i]
                self.n_vars = int(line)
                continue
            if line[0].isalpha():
                if (
                    line[0] == "v"
                ):  # variable assignments as positive or negative integers
                    for s in line[1:].strip().split():
                        i = int(s)
                        if i > 0:
                            self.b_values[i] = True
                        if i < 0:
                            self.b_values[-i] = False
                else:
                    pass

            elif line.find("--") == 0:  # some awkward comment
                pass
            else:  # the line contains a clause (integers ended by 0)
                clause = []
                for s in line.split():
                    i = int(s)
                    if i == 0:
                        break  # end of clause
                    clause.append(i)
                self.add_clause(clause)

    def number_of_vars(self):
        return self.n_vars

    def number_of_clauses(self):
        return len(self.i_clauses)

    def clauses(self):
        return self.i_clauses

    def add_clause(self, clause):
        self.add_clauses([clause])

    def add_clauses(self, clauses):
        for c in clauses:
            # if the clause contains index abs(i) > nVars, update nVars
            for i in c:
                self.n_vars = max(self.n_vars, abs(i))
            # append the new clause:
            self.i_clauses.append(c)

    def set_value(self, i, value):
        self.b_values[abs(i)] = value

    def set_values(self, dict_of_values):
        for i in dict_of_values:
            self.set_value(i, dict_of_values[i])

    def get_value(self, i):
        return self.b_values[abs(i)]

    def is_satisfiable(self):
        for i in range(1, self.n_vars + 1):
            if i not in self.b_values:
                raise Exception(
                    "Not all variables have values. Variable " + str(i) + " does not."
                )

        for c in self.clauses():
            clause_sat = False
            for i in c:
                if (i > 0 and self.get_value(i)) or (i < 0 and not self.get_value(-i)):
                    clause_sat = True
                    break  # the clause
            if not clause_sat:
                return False

        return True  # here all clauses have been satisfied

    def store(self, *comments):
        f = open(self.filename, "w")
        for c in comments:
            f.write("c " + c + "\n")
        f.write(
            "p cnf "
            + str(self.number_of_vars())
            + " "
            + str(self.number_of_clauses())
            + "\n"
        )

        for c in self.clauses():
            s = ""
            for i in c:
                s += str(i) + " "
            s += "0"
            f.write(s + "\n")
        f.close()

    def __str__(self):
        res = ""
        res += (
            "p cnf "
            + str(self.number_of_vars())
            + " "
            + str(self.number_of_clauses())
            + "\n"
        )

        for c in self.clauses():
            s = ""
            for i in c:
                s += str(i) + " "
            s += "0"
            res += s + "\n"
        return res


if __name__ == "__main__":  # test
    cnf = DimacsFile("new.cnf")
    cnf.load()
    sol = DimacsFile("solution14351.txt")
    # sol = DimacsFile("solution2.txt")
    sol.load()
    sol.add_clauses(cnf.clauses())
    b = sol.is_satisfiable()
    print("satisfiable=" + str(b))
    exit(0)
    df = DimacsFile("in.cnf")
    df.load()
    print(df.number_of_vars(), df.number_of_clauses(), df.clauses())

    df2 = DimacsFile("out.cnf", df.number_of_vars(), df.clauses())
    df2.add_clause([100, -99])
    df2.store("added clause [100, 99]", "the number of vars should increase to 100")
