def remove_unused_vars(nvars, clauses):
    used_vars = set()
    n = 0
    max_v = 0
    for clause in clauses:
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v > max_v:
                max_v = v
            if v not in used_vars:
                used_vars.add(v)
                n += 1
    if n == nvars and max_v == n:
        return nvars, clauses  # do not change since all the variables are used
    # otherwise not all variables are used (or the wrong number specified)

    n = 0
    d = {}
    new_clauses = []
    for clause in clauses:
        new_clause = []
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v in d:
                new_v = d[v]
            else:
                n += 1
                new_v = n
                d[v] = new_v
            if lit > 0:
                new_clause.append(new_v)
            else:
                new_clause.append(-new_v)
        new_clauses.append(new_clause)

    return n, new_clauses


def remove_useless_clauses(clauses):
    """ Removes clauses and variables if variable only appears in clause with single element
    """
    var_n = max([abs(l) for c in clauses for l in c])
    var_count = [0] * (var_n + 1)
    max_c = -1
    for clause in clauses:
        max_c = max(len(clause), max_c)
        for lit in clause:
            var_count[abs(lit)] += 1

    if max_c <= 1:
        return clauses

    def remove_clauses(clause):
        if len(clause) == 1 and var_count[abs(clause[0])] <= 1:
            return False

        return True

    return list(filter(remove_clauses, clauses))
