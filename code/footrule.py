import utils
import time
import numpy as np

from scipy.optimize import linear_sum_assignment


ALGORITHM_NAME = "FootRule+"

def run(data, params):
    """
    This method implements the foorule+ algorithm by considering
    the input data and params to create an implicit bipartite graph
    representation and finding assigning each alternative to some
    rank in the final full rank sigma.
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

    # get the n x n np.array cost matrix by calling helper function
    # rows are the candidates and cols are the positions
    cost = createCostMatrix(data, n, N)

    # run optimization algorithm from scipy library
    _, ranking = linear_sum_assignment(cost)
    # convert answer from np.array to tuple format

    # TODO:correction factor because we use zero-indexing in createCostMatrix
    off_by_one = np.ones(np.shape(ranking), dtype=int)
    sigma = tuple(ranking + off_by_one)

    time_elapsed = (time.process_time() - start_time) * 1000

    #print("{ALGORITHM_NAME}: {sigma}\n")
    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed



def createCostMatrix(data, n, N):
    """
    This function creates a cost matrix based on the input top-lists and
    the rule: C(i, j) := sum_{r=1}^j  (j −r) · p(πi = r)
    --------------------------------------

    Params

    'data': Counter object 
            The keys  in this Counter are tuple top-lists and the 
            values are the mulitiplicities of each top-list. Both 
            the elements in the tuples and the values are ints

    'n': int
         The number of candidates, which is also the number of ranks

    'N': int
         The total number of voters
    ---------------------------------------

    Returns 
        'arr': 2D n x n np.array
               The cost matrix with candidates for the rows and ranks
               for the columns
        
    """
    # gets n by n matrix of occurances for each alternative for each rank
    p = utils.alternativeRankFrequency(data,n) / N
    
    # creates empty n by n cost matrix
    # should not be any empty entries by end of function
    arr = np.empty((n,n))

    # follows rule above and assigns cost C(i,j)
    for i in range(n):          # for each alternative
        for j in range(n):      # for each rank
            summ = 0                # start sum
            for r in range(j):
                summ += (j-r) * p[i, r]     # following algorithm formula
            arr[i,j] = summ

    ##alternatives that do not get ranked at all must have infinite cost
    #infcostalts = utils.unrankedAlternatives(p)
    #print(infcostalts)

    ##TODO: assigns infinite cost for all candidates that were not ranked by any
    ## top list in the input
    #infrow = np.ones((n,)) 
    #for alt in infcostalts:
    #    arr[alt-1] = infrow

    return arr


