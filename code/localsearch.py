import time
import utils
import numpy as np

ALGORITHM_NAME = "Local-Search"

def run(data, params, sigma=None):
    """
    Implements Local Search for Linear Assignment (Kemeny) Algorithm
    """

    # start time
    start_time = time.process_time()

    # get data statistics/params
    n = params['n']
    N = params['N']
    s0 = params['s0']


    # if sigma is not given by previous algorithm, 
    # make it a random starting permutation
    if sigma is None:
        sigma = np.random.permutation(n)

    precedenceMatrix = utils.precedenceMatrix(data,n)


    
    def bestMove(index):
        # set new_index default value to be -1
        new_index = -1

        # get cost at current index 
        # (to ensure that a new_index 
        # selected below does not tie the 
        # cost of leaving a candidate 
        # unmoved)
        oldCost = utils.disagreements(sigma,index,index,precedenceMatrix)
        min_cost = oldCost
        
        for i in range(n):
            curr_cost = utils.disagreements(sigma, index, i, precedenceMatrix)

            # if better than before, update
            if curr_cost < min_cost:
                new_index = i
                min_cost = curr_cost

        return new_index if new_index != index else -1



    while(True):
        # iterate over positions i randomly
        order = np.random.permutation(n)
        # flag to be used
        moved = False

        for i in order:
            b = bestMove(i)
            if b < 0:
                continue

            # so while loop runs again
            moved = True
            # retrieve candidate
            cand = sigma[i]
            # move candidate from pos i to pos b
            sigma = np.delete(sigma, i)
            sigma = np.insert(sigma, b, cand)
            

        # if not a single candidate was moved, exit loop
        if not moved:
            break

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma
    
