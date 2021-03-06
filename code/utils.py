import numpy as np
import heapq 
import itertools

def generalizedKendallTauDistance(data, sigma, n, N, s0=None):
    """
    This method computes the average Kendall Tau Distance by computing
    the generalized Kendall Tau Distance between each top-list in data 
    and sigma, then dividing the total by N.
    Before making comparisons between some top-list pi_i and sigma, we
    convert pi_i to the full ranking tau_i which is just pi_i but takes
    on sigma's candidates ordering for all candidates that are tied in
    pi_i (i.e. candidates that are not ranked by pi_i)

    --------------------------------

    Params

    'data': Counter object 
            The keys  in this Counter are tuple top-lists and the 
            values are the mulitiplicities of each top-list. Both 
            the elements in the tuples and the values are ints

    'sigma': int tuple
             A single full ranking returned by some algorithm
             being tested

    'n': int
         Number of alternatives/candidates. Also the number of ranks
         in a full ranking context

    'N': int
         The total number of voters in this instance dataset

    's0': int tuple
          The true ground ranking of a given instance dataset
          taken from some distribution (Mallows Model). We additionaly
          compute the regular KT distance between sigma and s0 to
          additionally compare how close different algorithms are to
          a general consensus.
    --------------------------------

    Returns

        'cost': float
                The generalized Kendall Tau Distance as defined by the 
                Simon and Mathieu paper 'How To Rank Top-Lists'

        'ground_dist': float [OPTIONAL]
                Another reasonable measure of accuracy: the distance
                between the approximate solution sigma and the ground
                truth (known optimal ranking s0). This is useful in
                cases where common prior are reasonable assumptions and
                has many application beyond voting theory

    """
    # sum_{i=1}^N K(sigma, tau_i) / N
    cost = 0
    for x in data:
        pi = piToTau(x, sigma)
        cost += kendall_tau(np.array(sigma), pi) * data[x]
    return cost / N


def kendall_tau(rank_a,rank_b):

    """Calculates the Kendall Tau distance.
    Keyword arguments:
        rank_a -- a ballot
        rank_b -- a ballot
    """
    tau = 0
    n_candidates = len(rank_a)
    
    aPos = dict()
    bPos = dict()
    for i in range(n_candidates):
        a_candidate = rank_a[i]
        aPos[a_candidate] = i

        b_candidate = rank_b[i]
        bPos[b_candidate] = i

    for i, j in itertools.combinations(range(n_candidates), 2):
        tau += (np.sign(aPos[i] - aPos[j]) ==
                -np.sign(bPos[i] - bPos[j]))

    return tau


def piToTau(pi, sigma):
    """
    Helper function that converts top-lists pi_i to tau_i full lists
    given some full list sigma that breaks ties for nonranked candidates
    in pi_i

    Params
        pi : tuple
             a top list
        sigma: tuple
               a full list

    Returns
        a (n,) np array, tau.
    """
    # set of the current top-list. Makes for fast containment queries
    pi_set = set(pi)

    # this list will contain all the tied elements in the order of sigma
    extension = []
    for e in sigma:
        if e not in pi_set:
            extension.append(e)

    # return the extended list pi + ties ordered based on sigma --> tau
    return np.array(list(pi) + extension)



def precedenceMatrix(data, n):
    """
    This functions computes the n by n precedence matrix 'q', where q[i,j] is the 
    number of top-lists for which candidate i is ranked before candidate j.
    Note that if candidate i precedes candidate j in some ranking, that means 
    i is preferred to j for the ranking.
    --------------------------------------

    Params

    'data': Counter object 
            The keys  in this Counter are tuple top-lists and the 
            values are the mulitiplicities of each top-list. Both 
            the elements in the tuples and the values are ints

    'n': int
         The number of candidates, which is also the number of ranks
    ---------------------------------------

    Returns 

        'q': 2D n x n np.array
            Precedence matrix specifying how often candidates 
            appear before other candidates.
    """
    q = np.zeros((n,n))
    allCandidates = {i for i in range(n)}

    for topList in data:
        rankedCandidates = set()
        for i in range(len(topList)):
            candidateOne = topList[i]
            rankedCandidates.add(candidateOne)

            for j in range(i+1,len(topList)):
                candidateTwo = topList[j]
                q[candidateOne,candidateTwo] += data[topList]

        # Every candidate in ranking x is now in
        # candidatesFromList. These candidates 
        # precede all unranked candidates, and 
        # the unranked candidates precede 
        # no one
        unrankedCandidates = allCandidates - rankedCandidates
        for unrankedCandidate in unrankedCandidates:
            for rankedCandidate in rankedCandidates:
                q[rankedCandidate, unrankedCandidate] += data[topList]

    return q



def disagreements(fullRanking, oldPosition, newPosition, precedenceMatrix):
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



def alternativeRankFrequency(data, n):
    """
    This functions computes an n by n matrix 'p' where p[i,j] is the number of
    voters that placed candidate i in rank j.
    --------------------------------------

    Params

    'data': Counter object 
            The keys  in this Counter are tuple top-lists and the 
            values are the mulitiplicities of each top-list. Both 
            the elements in the tuples and the values are ints

    'n': int
         The number of candidates, which is also the number of ranks
    ---------------------------------------

    Returns 

        'p': 2D n x n np.array
             Matrix describing the frequency of each alternative appearing in a
             given rank position
    """

    # initialize n by n array for occurence of each alternatives on each rank
    # the rows are the alternatives and the columns are the ranks
    # note: if no candidate ever appears in some rank, then then stays 0
    p = np.zeros((n,n))

    # loop over all the keys (top-lists) in data
    for x in data:
        # loop over each alternative index in a given top-list
        for i in range(len(x)):
            alternative = x[i]

            if type(alternative) != int:
                alternative = int(alternative)
            
            p[alternative, i] += data[x]

    # remark: p is zero indexed for rankings and alternatives.
    return p 




def scores(data, n, N):
    """
    Computes the probability that a candidate is ranked
    ----------------------------

    Params
        Same as above
    ----------------------------

    Returns
        A (n,) numpy array corresponding to the score of each candidate
    """
    return np.sum(alternativeRankFrequency(data,n), axis=1) / N




def unrankedAlternatives(data, n, N):
    """
    Given a rank frequency matrix, returns a tuple of the candidates that were
    not ranked by any top-list in the input
    --------------------------

    Params
    'rankfreq': (n,n) np.array
                Matrix desribing the frequency of eacch alternative appearing in
                a given rank position
    --------------------------

    Returns
        A tuple of the alternatives that never appear in any top-list
    """
    off_by_one =  np.where(np.isclose(scores(data,n,N), 0))[0].tolist()
    return tuple(off_by_one)




def avgRanks(data, n, N):
    """
    Computes the average rank for each candidate conditioned on them appearing
    at least in one input list. Rank for candidate i is defined as
        Rank_i := sum_{r=1}^n p(pi_i=r) / Score_i · r 
    ----------------------------

    Params
        Same as above
    ----------------------------

    Returns
        A (n,) np.array of each candidate's average rank. A float('inf') is for
        candidates that never appear in the input list
    """
    # get scores of all candidates
    sc = scores(data, n, N)
    # get rank frequency for all candidates (and convert to prob by / N)
    freqs = alternativeRankFrequency(data,n) / N
    # [1, 2, 3, ..., n] used to multiply sum element
    r = np.arange(1, n+1)

    #initialize empty 1D (n,) np.array
    ranks = np.empty((n,))

    for i in range(n):
        # if candidate doesn't even appear once, rank is infinity
        if sc[i] == 0:
            ranks[i] = float('inf')
        # formula above
        else:
            ranks[i] = np.sum((freqs[i] / sc[i]) * r)
        
    return ranks




def lineGenerator(length):
    line = ""
    for i in range(length):
        line += "-"
    return line
