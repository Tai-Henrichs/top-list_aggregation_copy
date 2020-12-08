import utils
import time
import numpy as np
import operator

from scipy.optimize import linear_sum_assignment


ALGORITHM_NAME = "RandomSort"

def run(data, params):
    """
    This method implements the random sort algorithm.
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

    sigma = {}
    for l, _ in orderToplists(data, N):
        for candidate in l:
            sigma.setdefault(candidate)
    sigma = tuple(sigma)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, round(utils.generalizedKendallTauDistance(data, sigma, n, N, s0), 10), round(time_elapsed, 10) 

def orderToplists(data, numVoters, seed=None):
    # Let l be a top-list key in data, and 
    # let data[l] be the frequency of l. Then, 
    # x is a randomly sampled element from the exponential
    # distribution with scale numVoters/f (equivalently with rate f/numVoters).
    # The function creates a list of tuples for each list L, each tuple taking
    # the following form: (l, x)
    # The resulting list of tuples is sorted in ascending order of the 
    # values of x. In the event of a tie between two top-lists A and B
    # w.r.t. their sampled values x, A and B will retain their order from 
    # data.
    #  Parameters
    #  ----------
    #  data: Counter object 
    #        The keys  in this Counter are tuple top-lists and the 
    #        values are the mulitiplicities of each top-list. Both 
    #        the elements in the tuples and the values are ints
    #  numVoters: int or float
    #        The total number of voters
    # seed: int
    #        Seed used for random number generation
    rng = np.random.default_rng(seed)

    # scale = 1 / (data[l] / numVoters) avoids divide by zero 
    # that numVoters / data[l] enounters when data[l] is zero
    topLists = [(l,rng.exponential(scale = 1 / (data[l] / numVoters))) for l in data]
    topLists.sort()
    return topLists
     


    





