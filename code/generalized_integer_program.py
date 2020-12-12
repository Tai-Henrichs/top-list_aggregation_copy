import numpy as np
import time
import utils
import pulp as plp 
import itertools

ALGORITHM_NAME = "OPTIMAL_SOLUTION"

def run(data, params, baseList=None, permBound=None):
    """
    Outputs the complete ranking 
    that minimizes average kendall-tau 
    distance to the top-lists in data 
    under the following constraint:
    only the first permBound items of 
    the full-ranking specified by baseList
    can be permuted.
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

    'baseList': list or tuple
            List that determines which items cannot be permuted.
            The baseList must be a full-ranking.
            See function description for further details. 

            If not provided, all possible full-rankings are 
            considered.
        
    'permBound': int
            Only the first permBound items of baseList 
            can be permuted. Must be at least one.

            If not provided, all possible full-rankings are 
            considered.
            
    ------------------------------------

    Returns 

    sigma: tuple
            The optimal full-ranking, given 
            the provided constraints

    """
    # 'n' is the number of candidates, also the number of ranks
    n = params['n']
    
    # Handle default arguments
    if permBound is None or permBound > n or baseList is None:
        permBound = n

    model = plp.LpProblem("Kemeny Integer Program")

    indices = tuple(i for i in range(permBound))
    indexPermutations = tuple(pair for pair in itertools.permutations(indices, r=2))
    indexCombinations = tuple(pair for pair in itertools.combinations(indices, r=2))
    
    separator = "_"

    # Overview: First, compute the top-permBound-list that has the minimum average kendall-Tau 
    # distance to the top-lists in data using integer-programming. Second, append the 
    # unpermutable portion of baseList onto the end of the top-permBound-list from step one.
    precedenceMatrix = utils.precedenceMatrix(data, len(indices))

    # x_{i,j} is a binary variable since the full-ranking either places 
    # candidate i before candidate j, or it does not
    x_vars = {(i,j) : plp.LpVariable(cat=plp.LpBinary,name=f"{i}{separator}{j}") for i,j in indexPermutations}

    # Set constraints

    # Any valid full-ranking has no ties, so we must have that 
    # x_{i,j} + x_{j,i} = 1
    # which means that either i precedes j, or j precedes i
    #
    # Uses combinations to avoid duplicate constraints
    for i, j in indexPermutations:
        model.addConstraint(plp.LpConstraint(
                            e=plp.LpAffineExpression(
                                [(x_vars[(i,j)], 1), (x_vars[(j,i)], 1)]),
                            sense=plp.LpConstraintEQ,
                            rhs=1,
                            name=f"Strict_ranking_{i}{separator}{j}"))

    # Enforce transitivity: if i precedes j, and j precedes k, i must precede k
    # Uses permutations because enforicing transitivity requires considering 
    # different potential orderings of i, j, and k relative to each other
    for i,j,k in itertools.permutations(indices, r=3):
        model.addConstraint(plp.LpConstraint(
                            e=plp.LpAffineExpression(
                                [(x_vars[(i,j)], 1), (x_vars[(j,k)], 1), (x_vars[(k,i)], 1)]),
                            sense=plp.LpConstraintGE,
                            rhs=1,
                            name=f"Transitivity{separator}{i}{separator}{j}{separator}{k}"))

    # Enforce permutation limits: the items that appear after the 

    # Define the objective function for Kemeny
    # If a list ranks j before i, then the contributed cost is the number 
    # of voters that ranked i before j, which is precedenceMatrix[i,j]
    # (and vice versa when i and j have swapped order)
    #
    # Uses combinations to avoid counting the cost of 
    # a given swap twice, which improves performance
    kendall_dist = plp.lpSum(precedenceMatrix[i,j] *
                                x_vars[j,i] + 
                                    precedenceMatrix[j,i] * 
                                        x_vars[i,j] 
                                        for i,j in indexCombinations)

    # Require Kemeny-distance to be minimized
    model.sense = plp.LpMinimize
    model.setObjective(kendall_dist)

    # msg = 0 suppresses log information
    model.solve(plp.PULP_CBC_CMD(msg=0))

    # Dictionary to track how many candidates a given candidate precedes
    precedenceFreqency = {i:0 for i in indices}

    for var in model.variables():
        # Process the string name of variable
        name = var.name
        i = int(name.split(separator)[0])

        # Update precedence frequency if i precedes a candidate j
        if var.varValue == 1:
            precedenceFreqency[i] += 1

    sigma = [-1 for i in indices]
    for candidate, frequency in precedenceFreqency.items():
        # In the final list, candidate must precede 
        # frequency candidates. One is subtracted since 
        # candidates do not precede themselves
        # 
        # For example, with 10 candidates,
        # the candidate that precedes everyone would precede 
        # 9 candidates since they don't precede themselves. 
        index = len(sigma) - frequency - 1
        sigma[index] = candidate
    
    # Append the fixed portion of baseList onto 
    # the optimal sigma, assuming some items of 
    # the baseList are fixed
    if permBound < n and baseList is not None:
        fixedElements = baseList[permBound:]
        sigma.extend(fixedElements)

    # Convert to tuple for consistency
    return tuple(sigma)
