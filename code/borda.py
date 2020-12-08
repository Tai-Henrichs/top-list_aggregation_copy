import numpy as np
import time
import utils

ALGORITHM_NAME = "Borda+"

def run(data, params):
    
    """
    Runs borda+ algorithm. Gets average rank for each candidate
    and sorts them.
    ------------------------------------
        
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
    # start timer
    start = time.process_time() 

    # get dataset params
    n = params['n']
    N = params['N']
    s0 = params['s0']

    # arg (index+1) sort by increasing avg rank 
    sigma = np.argsort(utils.avgRanks(data, n, N)) + np.ones((n,)) 
    
    # end timer
    time_elapsed = (time.process_time() - start) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed


