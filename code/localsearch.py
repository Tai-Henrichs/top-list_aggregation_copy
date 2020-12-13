import time
import utils
import numpy as np

ALGORITHM_NAME = "Local-Search"

def run(data, params, sigma=None):

    # start time
    start_time = time.process_time()

    # get data statistics/params
    n = params['n']
    N = params['N']
    s0 = params['s0']


    # if sigma is not given by previous algorithm, 
    # make it a random starting permutation
    if sigma == None:
        sigma = np.random.permutation(n)

    precedenceMatrix = utils.precedenceMatrix(data,n)

    def disagreements(fullRanking, oldPosition, newPosition):
        """
        Let R be the ranking that results from  placing 
        the candidate at oldPosition in fullRanking at 
        newPosition, still within fullRanking. 
        Return the number of pairwise disagreements 
        between R and top-lists in data
        concerning the rank of 'candidate'. 
        --------------------------------------

        Params

        'fullRanking': list or tuple of ints
                A ranking of all candidates.

        'oldPosition': int
                Index into fullRanking 
                that stores the candidate being 
                swapped.

        'newPosition': int
                Index where the candidate 
                fullRanking[oldPosition]
                is to be moved for 
                computing pairwise disagreements.
        ---------------------------------------

        Returns int
        """
        disagreements = 0
        candidate = fullRanking[oldPosition]

        # Count top-lists that place candidates 
        # preceeding candidate in the given 
        # partialRanking after candidate
        for i in range(newPosition):
            otherCandidate = fullRanking[i]
            # otherCandidate may equal candidate (if newPosition > oldPosition), 
            # but then precedenceMatrix[candidate,otherCandidate] will be 0
            disagreements += precedenceMatrix[candidate,otherCandidate]

        # Count top-lists that place candidates 
        # ranked after candidate in the given 
        # partialRanking before candidate
        for i in range(newPosition, len(fullRanking)):
            otherCandidate = fullRanking[i]
            # otherCandidate may equal candidate (if newPosition < oldPosition), 
            # but then precedenceMatrix[candidate,otherCandidate] will be 0
            disagreements += precedenceMatrix[otherCandidate,candidate]

        return disagreements
    
    def bestMove(index):
        # set new_index default value to be -1
        new_index = -1

        # get cost at current index 
        # (to ensure that a new_index 
        # selected below does not tie the 
        # cost of leaving a candidate 
        # unmoved)
        oldCost = disagreements(sigma,index,index)
        min_cost = oldCost
        
        for i in range(n):
            curr_cost = disagreements(sigma, index, i)

            # if better than before, update
            if curr_cost < min_cost:
                new_index = i
                min_cost = curr_cost

        return new_index if new_index != index else -1


    while(True):
        # iterate over positions i randomly
        order = np.random.permutation(n)
        # flag to be used
        moved = False

        for i in order:
            b = bestMove(i)
            if b < 0:
                continue

            # so while loop runs again
            moved = True
            # retrieve candidate
            cand = sigma[i]
            # move candidate from pos i to pos b
            sigma = np.delete(sigma, i)
            sigma = np.insert(sigma, b, cand)
            

        # if not a single candidate was moved, exit loop
        if not moved:
            break

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed, sigma
    
