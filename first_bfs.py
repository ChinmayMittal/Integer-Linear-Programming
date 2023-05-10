import sys
import numpy as np

EPSILON = 1e-10

def print_matrix(matrix):
    np.savetxt(sys.stdout, matrix, fmt='%.2f', delimiter='  ')
    
def read_input(filename):
    with open(filename, "r") as f:
        n, m = tuple(map(int, f.readline().strip("\n").split(" ")))
        print(f"N: {n}, M: {m}")
        b = list(map(int, f.readline().strip("\n").split(" ")))
        ## convert to minimization problem by inverting c
        c = list(map(lambda x : -1*int(x), f.readline().strip("\n").split(" ")))
        print(f"B: {b}")
        print(f"C: {c}")
        A = []
        for constraint_idx in range(m):
             A.append(list(map(int, f.readline().strip("\n").split(" "))))
        print(f"A: {A}")
        return n, m, b, c, A

def construct_tableau(n, m, b, c, A):
    tableau =  np.zeros(shape=(m+1, n+1+m), dtype=np.float128) 
    tableau[1:, 0] = np.array(b)  ### initial basic variables are the slack variables
    tableau[1:m+1, 1:n+1] = np.array(A) ### original constraints
    tableau[1:m+1, n+1:] = np.identity(m) ### appended slack constraints
    tableau[0, 1:] = np.array(c + [0]*m)
    return tableau       
             
def simplex(tableau):
    tableau = np.round(tableau, 7)
    optimal_found = not np.any(tableau[0, 1:] < 0.0 ) ### is any reduced cost < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        pivot_column_idx = np.argmax(tableau[0, 1:] < 0.0) + 1
        div_array = tableau[1:, 0]/tableau[1:,pivot_column_idx]
        div_array = np.round(div_array, 7)
        div_array[tableau[1:,pivot_column_idx]<0] = np.inf
        pivot_row_idx = np.where(np.logical_and(div_array>0, div_array==np.amin(div_array[div_array >=0])))[0][0]+1
        print(f"Pivot Idx: ({pivot_row_idx}, {pivot_column_idx})")
        tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != pivot_row_idx):
                tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]
        iterations += 1
        tableau = np.round(tableau, 7)
        print(f"Table After {iterations} iterations of Primal Tableau")
        print_matrix(tableau)
        optimal_found = not np.any(tableau[0, 1:] < 0.0 )
        
    return tableau
      
def dual_simplex_method(tableau):
    tableau = np.round(tableau, 7)
    optimal_found = not np.any(tableau[1:, 0] < 0.0 ) ### is any primal basic variable < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        print("REACHED HERE 1")
        pivot_row_idx = np.argmax(tableau[1:, 0] < 0.0) + 1
        print(pivot_row_idx)
        print("REACHED HERE 2")
        div_array = -1* tableau[0, 1:]/(tableau[pivot_row_idx, 1:]+EPSILON)
        print("REACHED HERE 3")
        div_array = np.round(div_array, 7)
        # print(tableau[0, 1:]/(tableau[pivot_row_idx, 1:]+EPSILON))
        # print(div_array)
        ##[TODO] rigrously check division by zero and other corner cases

        pivot_column_idx = np.where(np.logical_and( div_array >0, div_array==np.amin(div_array[div_array >0])))[0][0]+1
        print("REACHED HERE 4")                
        print(f"Pivot Idx ({pivot_row_idx}, {pivot_column_idx})")
        print("REACHED HERE 5")                
        tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != pivot_row_idx):
                tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]        
        iterations += 1
        tableau = np.round(tableau, 7)
        print(f"Table After {iterations} iterations of Dual Simplex")
        print_matrix(tableau)
        optimal_found = not np.any(tableau[1:, 0] < 0.0 )
        
    
    
    return tableau  
def gomory_helper(tableau, n):

    tableau = np.round(tableau, 7)
    basic_variables = tableau[1:, 0]
    
    is_integer = np.allclose(basic_variables, np.round(basic_variables))
    while( not is_integer):
        basic_variables = tableau[1:, 0]
        ## find the constraint row which does not have an integer basic solution
        print(f"About to chose constraint_idx:{np.round(np.modf(basic_variables)[0], 7) != 0}\n {np.modf(basic_variables)[0]}")
        constraint_idx = np.argmax(np.round(np.modf(basic_variables)[0], 7) != 0 )+1 ### row number of constraint which will generate the new constraint
        print(constraint_idx)
        new_row = (tableau[constraint_idx, :]) - np.floor(np.round(tableau[constraint_idx, :], 7)) ## represents the new constraints row
        tableau = np.vstack((tableau, -1*new_row)) ### add the new row
        new_column = np.zeros((tableau.shape[0], 1))
        new_column[-1] = 1
        tableau = np.hstack((tableau, new_column))
        print()
        print_matrix(tableau)
        
        print("Dual Simplex Method")
        tableau = dual_simplex_method(tableau)
        
        basic_variables = tableau[1:, 0]
        is_integer = np.allclose(basic_variables, np.round(basic_variables))
    
    ### find solutions
    solution = np.zeros(n)
    for variable_idx in range(n):
        ## check if variable is basic
        column = tableau[1:, variable_idx+1]
        if( np.max(column) == 1 and np.count_nonzero(column) == 1):
            idx_of_one = np.argmax(column)+1
            solution[variable_idx] = tableau[idx_of_one, 0]
    return solution        
    

def gomory(filename):
    n, m, b, c, A = read_input(filename)
    tableau = construct_tableau(n, m, b, c, A)
    print("Initial Tableau")
    print_matrix(tableau)
    relaxed_lp_optimal_tableau = simplex(tableau)
    solution =  gomory_helper(relaxed_lp_optimal_tableau, n)
    solution = solution.astype(int)
    print(solution)
    
gomory("data5.txt")
