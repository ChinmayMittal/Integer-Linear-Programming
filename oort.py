import ortools
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('SAT')
infinity = solver.infinity()

x = solver.IntVar(0.0, infinity, 'x')
y = solver.IntVar(0.0, infinity, 'y')
z = solver.IntVar(0.0, infinity, 'z')
w = solver.IntVar(0.0, infinity, 'w')
u = solver.IntVar(0.0, infinity, 'u')


solver.Add(3*x + 4*y + 2*z + 5*w <= 7)
solver.Add(-x + 5*y - z + 3*w -2*u <= 4)
solver.Add(2*y - w - u <= 1)
solver.Maximize(4*x + 8*y + 3*z + 10*w - u)


status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())
    print('z =', z.solution_value())
    print('w =', w.solution_value())
    print('u =', u.solution_value())
else:
    print('The problem does not have an optimal solution.')
