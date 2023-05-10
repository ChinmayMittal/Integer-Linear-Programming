import ortools
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('SAT')
infinity = solver.infinity()

x = solver.IntVar(0.0, infinity, 'x')
y = solver.IntVar(0.0, infinity, 'y')


solver.Add(2*x + 14*y <= 35)
solver.Add(2*x<= 7)
solver.Maximize(x + 10*y)


status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())
else:
    print('The problem does not have an optimal solution.')
