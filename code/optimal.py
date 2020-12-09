import numpy as np
import time

from mallows_kendall import kendall_tau
from itertools import permutations
from utils import piToTau

ALGORITHM_NAME = "OPTIMAL_SOLUTION"

def run(data, params):
    """
    Brute Force Kemeny-Young optimal rank aggregation.
    Note: this solution is !n so only feasible when n < 25
    Note: In order to use full-list brute force, we transform
    top-lists to full lists in the same way as described by
    mathieu paper (i.e. pi --> tau by keeping all elements of pi
    and concatenating simga / pi such that tau = pi + sigma /pi
    in the order of sigma)

    Credit to:
    https://github.com/btrevizan
    ---------------------

    Params
        Same as above
    ---------------------

    Returns
        algorithm label
        avg kendall_tau distance for optimal solution
        time elapsed

    
    """
    start_time = time.process_time()

    min_dist = np.inf
    best_rank = None

    N = params['N']
    n = params['n']


    for rank in permutations(range(1, n+1)):

        dist = np.sum((kendall_tau(piToTau(ballot, rank), np.array(rank)) * data[ballot]) for ballot in data)

        if dist < min_dist:
            min_dist = dist
            best_rank = rank

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, min_dist / N, time_elapsed
