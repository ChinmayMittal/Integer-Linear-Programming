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
        print(b)
        ## convert to minimization problem by inverting c
        c = list(map(lambda x : -1*int(x), f.readline().strip("\n").split(" ")))
        print(f"B: {b}")
        print(f"C: {c}")
        A = []
        for constraint_idx in range(m):
             A.append(list(map(int, f.readline().strip("\n").split(" "))))
        print(f"A: {A}")
        return n, m, b, c, A

def construct_tableau_2(n, m, b, c, A):
    # plus m for yi's
    tableau =  np.zeros(shape=(m+1, n+1+m), dtype=np.float128)
    tableau[1:, 0] = np.array(b)  ### initial basic variables are the slack variables
    tableau[1:m+1, 1:n+1] = np.array(A) ### original constraints
    tableau[1:m+1, n+1:] = np.identity(m) ### appended slack constraints

    for constraint_idx in range(1, m+1):
        if np.round(tableau[constraint_idx, 0], 7) < 0:
            tableau[constraint_idx, :] *= -1



    # append yi's

    extra = np.vstack((np.zeros(m), np.identity(m)))
    tableau = np.hstack((tableau, extra))
    A_new = tableau[1:, 1:]
    # define new reduced cost vector, c^T = cT - cbTAb^-1A
    c = np.zeros(shape=(n+m+m), dtype=np.float128)
    cb = np.ones(shape=(m), dtype=np.float128)
    c[n+m:] = np.ones(shape=(m), dtype=np.float128)
    # Ab^-1 is just identity
    c_red = c - cb@A_new
    tableau[0, 0] = -np.sum(tableau[1:, 0])
    tableau[0, 1:] = c_red
    base_indices = np.zeros(m)
    for i in range(0, m):
        base_indices[i] = 1+i+n+m

    # tableau = np.round(tableau, 7)
    return tableau, base_indices

def first_tableau(tableau, base_indices, n, m, c):
    
    while True:
        pivot_column_idx = -1
        pivot_row_idx = -1
        for i, index in enumerate(base_indices):
            if index >= m + n + 1:
                pivot_row_idx = i+1
                break
        if pivot_row_idx == -1:
            break

        for i in range(1, m+n+1):
            # not a redundant constraint
            if np.round(tableau[pivot_row_idx, i], 7) != 0:
                pivot_column_idx = i
                break
        
        if pivot_column_idx == -1:
            # remove the lth row
            print_matrix(tableau)
            print(f"Found redundant constraint")
            tableau = np.delete(tableau, (pivot_row_idx), axis=0)
            base_indices = np.delete(base_indices, (pivot_row_idx-1))
            m-=1
            # TODO REDUNDANT CONSTRAINT
            
        else:
            # remove the pivot row and add the pivot column

            base_indices[pivot_row_idx-1] = pivot_column_idx            
            tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
            for row_idx in range(tableau.shape[0]):
                if(row_idx != pivot_row_idx):
                    tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]
            


    # remove the excess columns
    # tableau = np.round(tableau, 7)    
    tableau = tableau[:, :m+n+1]
    
    # recompute reduced costs and original cost
    c_new = np.hstack((c, np.zeros(m)))
    cb = np.zeros(m)
    A  = tableau[1:, 1:]
    Ab = np.zeros((m, m))

    for i, index in enumerate(base_indices):
        Ab[:, i] =A[: , int(index)-1]
        cb[i] = c_new[int(index)-1]
    c_red = c_new - cb@np.linalg.inv(Ab)@A

    tableau[0][0] = -cb@tableau[1:, 0]
    tableau[0][1:] = c_red
    # tableau = np.round(tableau, 7)
    return tableau, base_indices
        

def construct_tableau(n, m, b, c, A):
    tableau =  np.zeros(shape=(m+1, n+1+m), dtype=np.float128) 
    tableau[1:, 0] = np.array(b)  ### initial basic variables are the slack variables
    tableau[1:m+1, 1:n+1] = np.array(A) ### original constraints
    tableau[1:m+1, n+1:] = np.identity(m) ### appended slack constraints
    tableau[0, 1:] = np.array(c + [0]*m)
    return tableau       
             
def simplex(tableau, base_indices):
    # tableau = np.round(tableau, 7)
    optimal_found = not np.any(np.round(tableau[0, 1:], 7) < 0.0 ) ### is any reduced cost < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        pivot_column_idx = np.argmax(np.round(tableau[0, 1:], 7) < 0.0) + 1
        div_array = tableau[1:, 0]/tableau[1:,pivot_column_idx]
        div_array[np.round(tableau[1:,pivot_column_idx], 7)<0] = np.inf
        pivot_row_idx = np.where(np.logical_and(np.round(div_array, 7)>=0, div_array==np.amin(div_array[np.round(div_array, 7) >=0])))[0][0]+1
        base_indices[pivot_row_idx-1] = pivot_column_idx
        print(f"Pivot Idx: ({pivot_row_idx}, {pivot_column_idx})")
        tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != pivot_row_idx):
                tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]
        iterations += 1
        # tableau = np.round(tableau, 7)
        print(f"Table After {iterations} iterations of Primal Tableau")
        print_matrix(tableau)
        optimal_found = not np.any(np.round(tableau[0, 1:], 7) < 0.0 )
        
    return tableau, base_indices
      
def dual_simplex_method(tableau, base_indices):
    # tableau = np.round(tableau, 7)
    optimal_found = not np.any(np.round(tableau[1:, 0], 7) < 0.0 ) ### is any primal basic variable < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        pivot_row_idx = np.argmax(np.round(tableau[1:, 0], 7) < 0.0) + 1
        div_array = -1* np.round(tableau[0, 1:], 7)/(np.round(tableau[pivot_row_idx, 1:], 7)+EPSILON)
        div_array[np.round(tableau[pivot_row_idx, 1:], 7)>=0] = np.inf
        div_array += EPSILON

        ##[TODO] rigrously check division by zero and other corner cases
        pivot_column_idx = np.where(np.logical_and( np.round(div_array, 7) >=0, div_array==np.amin(div_array[np.round(div_array, 7) >=0])))[0][0]+1
        print(f"Pivot Idx ({pivot_row_idx}, {pivot_column_idx})")
        base_indices[pivot_row_idx-1] = pivot_column_idx        
        tableau[pivot_row_idx, :] /= tableau[pivot_row_idx][pivot_column_idx]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != pivot_row_idx):
                tableau[row_idx, : ] -= tableau[pivot_row_idx, :] * tableau[row_idx][pivot_column_idx]        
        iterations += 1
        # tableau = np.round(tableau, 7)
        print(f"Table After {iterations} iterations of Dual Simplex")
        print_matrix(tableau)
        optimal_found = not np.any(np.round(tableau[1:, 0], 7) < 0.0 )
        
    
    
    return tableau, base_indices
def gomory_helper(tableau, n, base_indices):
    basic_variables = tableau[1:, 0]
    
    is_integer = np.allclose(basic_variables, np.round(basic_variables))
    while( not is_integer):
        basic_variables = tableau[1:, 0]
        ## find the constraint row which does not have an integer basic solution
        constraint_idx = np.argmax(np.modf(np.round(basic_variables, 7))[0] != 0 )+1 ### row number of constraint which will generate the new constraint
        new_row = (tableau[constraint_idx, :]) - np.floor(np.round(tableau[constraint_idx, :], 7)) ## represents the new constraints row
        tableau = np.vstack((tableau, -1*new_row)) ### add the new row
        new_column = np.zeros((tableau.shape[0], 1))
        new_column[-1] = 1
        base_indices = np.hstack((base_indices, np.array([len(tableau[0])])))
        tableau = np.hstack((tableau, new_column))

        print("Dual Simplex Method")
        tableau, base_indices = dual_simplex_method(tableau, base_indices)
        
        basic_variables = tableau[1:, 0]
        is_integer = np.allclose(basic_variables, np.round(basic_variables))
    
    ### find solutions
    solution = np.zeros(n)
    for variable_idx in range(n):
        ## check if variable is basic
        if (variable_idx+1) in base_indices:
            column = tableau[1:, variable_idx+1]
            idx_of_one = np.argmax(column)+1
            solution[variable_idx] = tableau[idx_of_one, 0]
    return solution        
    

def gomory(filename):

    # read file
    n, m, b, c, A = read_input(filename)
    # dual phase simplex
    # Ist phase
    print("Getting the tableau for first phase of dual phase simplex")
    tableau, base_indices = construct_tableau_2(n, m, b, c, A)
    # apply simplex on the current tableau
    print("Applying simplex on the large tableau, to get the corresponding tableau for second phase")
    print_matrix(tableau)
    tableau, base_indices = simplex(tableau, base_indices)
    if np.round(tableau[0][0], 7) != 0:
        print(f"No feasible solution exists")
        return
    else:

        tableau, base_indices = first_tableau(tableau, base_indices, n, m, c)

        # 2nd phase of simplex
        print("Applying simplex")
        relaxed_lp_optimal_tableau, base_indices = simplex(tableau, base_indices)
        # applying gomory cut method on the optimal primal for rlp
        print("Applying gomory cut")
        solution =  gomory_helper(relaxed_lp_optimal_tableau, n, base_indices)

        solution = solution.astype(int)
        print(solution)

    # print("Getting the tableau for first phase of dual phase simplex")
    # tableau= construct_tableau(n, m, b, c, A)
    # base_indices = np.zeros(m)
    # relaxed_lp_optimal_tableau, base_indices = simplex(tableau, base_indices)
    # print("Applying gomory cut")
    # solution =  gomory_helper(relaxed_lp_optimal_tableau, n)
    # solution = solution.astype(int)
    # print(solution)
    

if __name__ == '__main__':    
    gomory('data12.txt')
