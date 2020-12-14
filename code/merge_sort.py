import time
import utils

ALGORITHM_NAME = "Insertion Sort"

def run(data, params):
    """
    This method implements Copeland's voting rule, which 
    orders candidates from most to least pairwise contest 
    wins. 
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

    # Order candidates by non-decreasing pair-wise contest wins 
    # (ascending order with lexicographic tie-breaking)
    precedenceMatrix = utils.precedenceMatrix(data, n)

    # Credits to Sayan-Paul for starter code for merge sort
    # See: https://github.com/Sayan-Paul/Sort-Library-in-Python/blob/master/sortlib.py
    def mergesort(ar):
        if len(ar)<=1:
            return ar
        middle=len(ar)/2
        left =ar[:middle]
        right=ar[middle:]
        left=mergesort(left)
        right=mergesort(right)
        res=merge(left,right)
        return res

    def merge(left,right):
        res=[]
        while len(left)+len(right):
            if len(left)*len(right):
                if precedenceMatrix[left[0],right[0]]<=precedenceMatrix[right[0],left[0]]:
                    res.append(left[0])
                    left=left[1:]
                else:
                    res.append(right[0])
                    right=right[1:]
            elif len(left):
                res.append(left[0])
                left=left[1:]
            elif len(right):
                res.append(right[0])
                right=right[1:]
        return res

    candidates = [i for i in range(n)]
    sortedCandidates = mergesort(candidates)

    sigma = tuple(sortedCandidates)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma