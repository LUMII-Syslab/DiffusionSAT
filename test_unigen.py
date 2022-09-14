from utils.sat import run_external_solver, run_unigen, build_dimacs_file

if __name__ == '__main__':
    dimacs = build_dimacs_file([[-1,2],[1,-2]], 2)

    is_sat, solution = run_unigen(dimacs)
    print("unigen")
    print(is_sat, solution)
    print("lingeling")
    is_sat, solution = run_external_solver(dimacs)
    print(is_sat, solution)
