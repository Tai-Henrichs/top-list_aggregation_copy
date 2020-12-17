import utils 
import time

ALGORITHM_NAME = "Copeland"

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

    # save results to avoid re-computation
    pairwiseVictores = [-1 for i in range(n)]
    def totalPairwiseVictories(i):
        if not pairwiseVictores[i] == -1:
            return pairwiseVictores[i]

        totalVictories = 0
        for j in range(n):
            # If candidates i and j beat each 
            # other an equal number of times, i and j 
            # each have a victory
            if precedenceMatrix[i,j] >= precedenceMatrix[j,i]:
                totalVictories += 1

        pairwiseVictores[i] = totalVictories
        return totalVictories

    candidates = [i for i in range(n)]
    candidates.sort(key=totalPairwiseVictories, reverse=True)

    sigma = tuple(candidates)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma