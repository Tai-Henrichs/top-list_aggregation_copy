import utils
import time
import numpy as np
import operator
import math


ALGORITHM_NAME = "Score-Then-Adjust"

def run(data, params, epsilon = 1):
    """
    This method implements the Score-Then-Adjust EPTAS.
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

    # Order candidates by non-increasing scores (descending order with lexicographic tie-breaking)
    candidateScores = utils.scores(data,n,N)
    candidates = [i for i in range(n)]
    candidates.sort(key=lambda i : candidateScores[i], reverse=True)

    permBound = (1 + (1.0 / epsilon)) * (params['k'] - 1)
    permBound = math.ceil(permBound)
    # Consider all possible permutations of the sorted list of candidates, 
    # only allowing the first permBound candidates to be shifted in their locations
    # Select the permutation that minimizes kendall-tau distance
    
    # Multiply by negative one because utils.permute maximizes the objective function
    # So, to minimize Kendall-Tau, we should maximize the inverse of Kendall-Tau
    invertedKentallTau = lambda l : -1.0 * utils.generalizedKendallTauDistance(data, l, n, N, s0)

    # permute returns a list containing (top = 1) tuple
    # This assigns the second element of the first tuple to 
    # sigma because that is the permutation that minimizes kendall-tau
    sigma = utils.permute(candidates, permBound, measure=invertedKentallTau, top=1)[0][1]

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma




     


    





