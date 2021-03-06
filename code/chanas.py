import utils
import time
import numpy as np

ALGORITHM_NAME = "Chanas"

def run(data, params, sigma=None):
    """
    Implements the Chanas Algorithm
    """
    start_time = time.process_time()

    # get data statistics/params
    n = params['n']
    N = params['N']
    s0 = params['s0']


    # if sigma is not given by previous algorithm, 
    # make it a random starting permutation
    if sigma is None:
        sigma = np.random.permutation(n)
    else:
        sigma = np.array(sigma)

    p_matrix = utils.precedenceMatrix(data,n)

    def sort(s):
        for i in range(len(s)):
            for j in range(i):
                old_cost = utils.disagreements(s,i,i,p_matrix)
                curr_cost = utils.disagreements(s,i,j,p_matrix)

                if curr_cost < old_cost:
                    # retrieve candidate
                    cand = s[i]
                    # move candidate from pos i to pos b
                    s = np.delete(s, i)
                    s = np.insert(s, j, cand)
                    break
        return s


    # first pass
    sigma = sort(sigma)
    # reverse
    sigma = np.flip(sigma)
    # second pass
    sigma = tuple(sort(sigma))
                
    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma
