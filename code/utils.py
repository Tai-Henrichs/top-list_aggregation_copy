import numpy as np
from mallows_kendall import kendall_tau
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
        #print(f'tau_x: {tau_x}')
        #print(f'sigma: {sigma}')
        cost += kendall_tau(np.array(x), np.array(sigma)) * data[x]

    return cost / N

def alternativeRankFrequency(data, n, N):
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
    return p / N


def unrankedAlternatives(rankfreq):
    """
    Given a rank frequency matrix, returns a tuple of the candidates that were
    not ranked by any top-list in the input
    """
    off_by_one =  np.where(np.all(np.isclose(rankfreq, 0), axis=1))[0].tolist()
    return tuple(i+1 for i in off_by_one)

