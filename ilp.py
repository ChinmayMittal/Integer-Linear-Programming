import sys
import numpy as np

def print_matrix(matrix):
    np.savetxt(sys.stdout, matrix, fmt='%.2f', delimiter=' ')
    
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
    optimal_found = not np.any(tableau[0, 1:] < 0 ) ### is any reduced cost < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        pivot_column_idx = np.argmax(tableau[0, 1:] < 0) + 1
        div_array = tableau[1:, 0]/tableau[1:,pivot_column_idx]
        pivot_row_idx = np.where(np.logical_and( div_array >= 0, div_array==np.amin(div_array[div_array >=0])))[0][0]+1
        print(f"Pivot Idx: ({pivot_row_idx}, {pivot_column_idx})")
        tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != pivot_row_idx):
                tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]
        iterations += 1 
        print(f"Table After {iterations} iterations")
        print_matrix(tableau)
        optimal_found = not np.any(tableau[0, 1:] < 0 )
        
    return tableau
        

def gomory(filename):
    n, m, b, c, A = read_input(filename)
    tableau = construct_tableau(n, m, b, c, A)
    print("Initial Tableau")
    print_matrix(tableau)
    relaxed_lp_optimal_tableau = simplex(tableau)

gomory("data2.txt")