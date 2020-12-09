import utils
import time
import numpy as np
import operator
import math
import pulp


ALGORITHM_NAME = "Integer-Program"

def run(data, params):
    """
    This method implements an exact algorithm for 
    finding the optimal solution to the Kemeny top-list problem
    using integer programming.
    -------------------------------------

    Params

    'data': Counter object 
            The keys  in this Counter are tuple top-lists and the 
            values are the mulitiplicities of each top-list. Both 
            the elements in the tuples and the values are ints

    'params': dict
              A python dictionary with that holds statics and info
              on the dataset stored on 'data'. Some of the keys are
              'n', 'N', 'k', and 'theta'. Refer to sim.py for full
              documentation.
    ------------------------------------

    Returns 

    'AGORITHM_NAME': str
                     The identifier for the algorithm implemented in
                     this current file.

    'time': float
            Time if took for the main component of the algorithm to run

    'acccuracy': float
                 The generalized Kendall Tau Distance between the 
                 dataset top-lists and sigma

    """
    start_time = time.process_time()

    # 'n' is the number of candidates, also the number of ranks
    n = params['n']
    # 'N' is the total number of voters
    N = params['N']
    # 's0' is the optional ground truth full ranking of the candidates
    # (distribution is drawn off this full ranking)
    s0 = params['s0']

    model = cpx.Model(name="Kemeny Integer Program")

    indices = range(0,n)

    precedenceMatrix = utils.precedenceMatrix(data, n)

    # x_{i,j} is a binary variable since the full-ranking either places 
    # candidate i before candidate j, or it does not
    x = model.binary_var_dict([(i,j) for i in indices for j in indices])

    # Any valid full-ranking has no ties, so we must have that 
    # x_{i,j} + x_{j,i} = 1
    model.add_constraints( (x[(i,j)] + x[(j,i)] == 1 for i in indices for j in indices) )
    
    # Enforce transitivity: if i precedes j, and j precedes k, i must precede k
    model.add_constraints( (x[(i,j)] + x[(j,k)] + x[(k,i)] >= 1 
                                for i in indices for j in indices for k in indices) )
    
    # Define the objective function for Kemeny
    # If a list ranks j before i, then the contributed cost is the number 
    # of voters that ranked i before j, which is precedenceMatrix[i,j]
    # (and vice versa when i and j have swapped order)
    kendall_dist = model.sum(precedenceMatrix[i,j] * x[(j,i)] + precedenceMatrix[j,i] * x[(i,j)] 
                                for i in indices for j in indices)

    # Require Kemeny-distance to be minimized
    model.minimize(kendall_dist)

    # Solve with local CPLEX installation
    solution = model.solve()

    # Reconstruct ranking from solution
    # Note that the solver can provdie the 
    # objective value, but reconstruction
    # is necessary for consistency with 
    # other algorithms
    sigma = []
    for var, value in solution.iter_var_values():
        print(f"Solution is {var} = {value}")

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed




     


    





