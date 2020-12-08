import numpy as np
from mallows_kendall import kendall_tau
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
    def piToTau(pi):
        # set of the current top-list. Makes for fast containment queries
        pi_set = set(pi)

        # this list will contain all the tied elements in the order of sigma
        extension = []
        for e in sigma:
            if e not in pi_set:
                extension.append(e)

        # return the extended list pi + ties ordered based on sigma --> tau
        return tuple(list(pi) + extension)

    
    # sum_{i=1}^N K(sigma, tau_i) / N
    cost = 0
    for x in data:
        tau_x = piToTau(x)
        cost += kendall_tau(np.array(x), np.array(sigma)) * data[x]

    return cost / N

def alternativeRankFrequency(data, n):
    """
    This functions computes an n by n matrix 'p' where p[i,j] is the number of
    voters that placed candidate i in rank k.
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
            # get alternative ID (and decrement for zero-index matrix)
            alternative = x[i] - 1
            #TODO: make all synthetic and real datasets type consists instead 
            # of doing this
            if type(alternative) != int:
                alternative = int(alternative)
            # make sure to increment by number of multiplicities of current top-list
            p[alternative, i] += data[x]

    # remark: p is zero indexed for rankings and alternatives. The necessary change of
    # decrementing all alternative IDs by one was made.
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
    return tuple(i+1 for i in off_by_one)



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

def permute(l, permBound, measure=None, top=None):
    """
    Considers all possible permutations of the items in l 
    which can be made by permuting the first permBound elements 
    of l. Returns a list of the permutations that maximize 
    measure, where the number of permutations 
    returned is specified by top. The returned permutations 
    are ordered from most to least satisfaction of measure. 
        ----------------------------
        Params
            l: list
            permBound: int
            measure: function with one parameter that receives a list
            top: int
        ----------------------------

        Returns a list of permutations, each represented by a list
    """
    fixedElements = l[- (len(l) - permBound)]

    # Case in which one (or both) of measure and top are None
    perms = []
    if top is None or measure is None:
        perms = [perm.append(fixedElements) for perm in itertools.permutations(l[:permBound])]

    if top is None:
        if measure is not None:
            perms.sort(key=measure)
        return perms

    if measure is None:
        return perms[:top]

    # Case in which both top and measure are specified 

    # bestPerms is a min-heap that stores the top permutations 
    # maximizing measure so far
    bestPerms = []
    for perm in itertools.permutations(l[:permBound]):
        perm = perm.append(fixedElements)
        perm = (measure(perm), perm)
        if not bestPerms or len(bestPerms) < top:
            heapq.heappush(bestPerms, perm)
        else:
            heapq.heappushpop(bestPerms,perm)
    bestPerms.sort(key=measure)
    return bestPerms








