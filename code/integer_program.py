import utils
import pulp as plp 
import itertools
import math

ALGORITHM_NAME = "OPTIMAL_SOLUTION"

def solve(data, params, lpRelaxation=False, baseList=None, permBound=None):
    """
    Outputs a full-ranking.
    
    If lpRexalaxation is False, the
    output full-ranking minimizes 
    distance to the top-lists in data 
    under the following constraint:
    only the first permBound items of 
    the full-ranking specified by baseList
    can be permuted.

    If lpRelaxation is True, 
    a non-optimal solution may be returned, as 
    a linear-programming relaxation of 
    the optimal integer-program will be 
    utilized.
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

    'lpRelaxation'" boolean
            True if a linaer programming relaxation should be 
            utilized, False otherwise. 

            If not provided, the exact integer-programming 
            solution will be utilized.

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
    if baseList is None:
        baseList = tuple(i for i in range(n))

    if permBound is None or permBound > n:
        permBound = n
    
    # Spaces are not supported by Pulp, use '_' 
    # to separate names instead
    # Otherwise, Pulp will generate warnings
    separator = "_"

    model = plp.LpProblem(f"Kemeny{separator}Integer{separator}Program")

    indices = baseList[:permBound]
    indexPermutations = tuple(pair for pair in itertools.permutations(indices, r=2))
    indexCombinations = tuple(pair for pair in itertools.combinations(indices, r=2))
    
    # Overview: First, compute the top-permBound-list that has the minimum average kendall-Tau 
    # distance to the top-lists in data using integer-programming. Second, append the 
    # unpermutable portion of baseList onto the end of the top-permBound-list from step one.
    
    # precedenceMatrix still n by n because, otherwise, 
    # there can be out-of-bound errors when constructing 
    # precedenceMatrix, and a candidate with a numeric 
    # label >= permBound is considered.
    precedenceMatrix = utils.precedenceMatrix(data, n)

    # x_{i,j} is a binary variable since the full-ranking either places 
    # candidate i before candidate j, or it does not

    # Using LpBinary works fine, too, but this way the bounds set in 
    # are not redundant
    variableType = plp.LpContinuous if lpRelaxation else plp.LpInteger

    x_vars = {(i,j) : plp.LpVariable(cat=variableType,
                                        lowBound=0,
                                        upBound=1,
                                        name=f"{i}{separator}{j}") for i,j in indexPermutations}

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
                            name=f"Strict{separator}ranking{separator}{i}{separator}{j}"))

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
        candidateLabels = name.split(separator)
        i = int(candidateLabels[0])

        value = var.varValue

        # Update precedence frequency if i precedes a candidate j
        #
        # Comparisons handle the case that linear programming was used
        if value >= .5:
            precedenceFreqency[i] += 1

    # save for reuse
    precedenceFreqencyIter = precedenceFreqency.items()

    sigma = [candidate for candidate, _ in precedenceFreqencyIter]
    if lpRelaxation:
        # If linear programming is used, 
        # we may have x_{i}_{j} = x_{j}_{i} = .5, 
        # causing precedenceFrequency[i] to equal 
        # precedenceFrequency[j]
        # Could address this with additional constraints,
        # but breaking ties in precedenceFrequency 
        # lexicographically is generally more efficient
        sigma.sort(key=lambda num : precedenceFreqency[num], reverse=True)
    else:
        for candidate, frequency in precedenceFreqencyIter:
            # In the final list, candidate must precede 
            # frequency candidates. One is subtracted since 
            # candidates do not precede themselves
            # 
            # For example, with 10 candidates,
            # the candidate that precedes everyone would precede 
            # 9 candidates since they don't precede themselves. 
            #
            # This only works because precedenceFrequency[i]
            # will never equal precedenceFrequency[j] for 
            # any i and j, which is why this solution is only 
            # used if integer-programming is used. Note that 
            # this is faster than the sorting solution used 
            # for linear-programming - O(n) vs. O
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
