import time
import utils
import quick_sort_base as qsb

ALGORITHM_NAME = "QS-Det"

def run(data, params):
    """
    This method implements a quick-sort 
    algorithm that uses a deterministic 
    pivot-selection. In particular, 
    each candidate in a list is considered 
    as a pivot, and the chosen pivot 
    is the candidate with the 
    fewest pair-wise disagreements respecting 
    candidates preceeding and following the 
    the list being divided by the pivot.
    
    During sorting, 
    candidate a > candidate b iff 
    a precedes b more often than the reverse
    in the top-lists provided in data.
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

    precedenceMatrix = utils.precedenceMatrix(data, n)

    def pivotCost(arr, start, end, pivot):
        cost = 0
        candidate = arr[pivot]

        for i in range(start, pivot):
            otherCandidate = arr[i]
            cost += precedenceMatrix[candidate, otherCandidate]

        for i in range(pivot, end+1):
            otherCandidate = arr[i]
            cost += precedenceMatrix[otherCandidate, candidate]

        return cost

    def bestPrecedence(arr, start, end):
        bestPivotSoFar = start
        minPivotCost = pivotCost(arr, start, end, start)

        for i in range(start+1,end+1):
            currCost = pivotCost(arr, start, end, i)
            if currCost < minPivotCost:
                bestPivotSoFar = i 
                minPivotCost = currCost

        return bestPivotSoFar

    candidates = [i for i in range(n)]
    qsb.quicksort(precedenceMatrix, candidates, bestPrecedence)

    sigma = tuple(candidates)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma


