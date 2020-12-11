import time
import utils
import borda
import numpy as np

ALGORITHM_NAME = "Score-Then-Borda+"

def run(data, params):
    """
    Runs score then borda algorithm. Creates partitions t:  E_0,..E_inf by using
    rough scores and random variable u. Then sorts by average rank within
    parition. Appends to output sigma partition by partiton in order or t
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
    start_time = time.process_time()

    # get dataset statistics
    n = params['n']
    N = params['N']
    s0 = params['s0']


    # random var (float) u [0,1)
    rng = np.random.default_rng(params['seed'])
    u = rng.uniform(0,1)
    # get scores array
    scores = utils.scores(data, n, N)
    # get avg ranks array
    avg_ranks = utils.avgRanks(data, n, N)

    # keep track of paritions using python dict
    partitions = {}
    for i in range(n):
        # get score for candidate i
        sc_i = scores[i]
        # if never appeared in any top-list then put them in E_inf
        if sc_i <= 0:
            t = float('inf')
        # else compute according to formula in algo description
        else:
            t = int(np.floor(u - np.log(sc_i))) 

        # add i to partition t
        if t in partitions:
            l = partitions[t] + [i]
            partitions[t] = l
        else:
            partitions[t] = [i]


    # helper method that applies borda rule within each partiton
    def bordaHelper(arr):
        # create tuple of avg (rank, candidate) pairs
        tups = [(avg_ranks[i], i) for i in arr]
        # sort by increasing avg rank
        tups.sort(key= lambda x: x[0])

        # only return candidates
        return [x[1] for x in tups]

    # create master list and append candidates from each parition
    # in order or increasing partition t
    master = list()
    for key in sorted(partitions):
        currlist = bordaHelper(partitions[key])
        # concatenate lists
        master += currlist

    # convert to tuple to stay consistent
    sigma = tuple(master)

    #end time
    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma




