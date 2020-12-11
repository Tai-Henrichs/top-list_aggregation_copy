import utils
import time
import numpy as np
import operator

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

    # Get all top-lists from data
    topLists = [t for t in data.keys()]

    def exponentialVarFromList(t):
        # t is a tuple from data representing a top-list

        # scale = 1 / (data[t] / numVoters) avoids divide by zero 
        # that numVoters / data[t] enounters when data[t] is zero
        # Note that numVoters should never be zero since that implies 
        # an empty data-set
        rng = np.random.default_rng(params['seed'])
        scale = 1 / (data[t] / N)
        return rng.exponential(scale) 

    # Sorts lists in ascending order of values sampled 
    # from the exponential distribution associated with 
    # each list. The exponential distribution associated 
    # with a given list has rate equal to 
    # f / N, where f is the frequency of the list 
    # in question
    topLists.sort(key=exponentialVarFromList)

    # Use a dictionary to improve performance 
    # when checking whether a candidate has 
    # already been added to the top-list
    # Relies on dictionaries being ordered 
    # as of Python 3.7
    sigma = dict()
    rankedCanidates = set()
    for l in topLists:
        for candidate in l:
            sigma.setdefault(candidate)
            rankedCanidates.add(candidate)
    
    # Finish by adding candidates that are never ranked
    allCandidates = {i + 1 for i in range(n)}
    unrankedCandidates = allCandidates - rankedCanidates

    for unrankedCandidate in unrankedCandidates:
        sigma.setdefault(unrankedCandidate)

    sigma = tuple(candidate for candidate in sigma)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma



    