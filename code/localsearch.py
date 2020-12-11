import time
import utils
import numpy as np

ALGORITHM_NAME = "Local-Search"

def run(data, params, sigma=None):

    # start time
    start_time = time.process_time()

    # get data statistics/params
    n = params['n']
    N = params['N']
    s0 = params['s0']


    # if sigma is not given by previous algorithm, 
    # make it a random starting permutation
    if sigma == None:
        sigma = np.random.permutation(n)

    # simply call to distance function
    def dist(s):
        s = tuple(s)
        return utils.generalizedKendallTauDistance(data, s, n, N, s0)


    def bestMove(index):
        # initialize min_cost to infinity
        min_cost = float('inf')
        # get candidate at index to be moved later, this doesn't change
        # during the execution of this helper function
        cand = sigma[index]

        # set new_index default value to be -1
        new_index = -1
        
        for b in range(n):
            # create the move
            new_sigma = np.copy(sigma)
            new_sigma = np.delete(new_sigma, index)
            new_sigma = np.insert(new_sigma, b, cand) 

            # compute the distance of such move
            curr_cost = dist(new_sigma)
            # if better than before, update. Note new_index will reupdate
            # to index if there does not exist any better move
            if curr_cost < min_cost:
                new_index = b
                min_cost = curr_cost

        return new_index if new_index != index else -1


    while(True):
        # iterate over positions i randomly
        order = np.random.permutation(n)
        # flag to be used
        moved = False
        curr_cost = dist(sigma)

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
        if not moved or dist(sigma) == curr_cost:
            break

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, dist(sigma), time_elapsed, sigma
    
