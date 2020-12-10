import utils
import time
import pulp as plp
import itertools


ALGORITHM_NAME = "Integer-Program"

def run(data, params):
    """
    This method implements an exact algorithm for 
    finding the optimal solution to the Kemeny top-list problem
    using integer programming.
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

    model = plp.LpProblem("Kemeny Integer Program")

    indices = tuple(i for i in range(n))

    # when setting the triples, should avoid k = i or k = j 
    # if the pair being used as the base is (i,j)
    # may want to change back to permutations and not product, maybe combos?
    pairs = tuple(pair for pair in itertools.permutations(indices, r=2))
    triples = tuple(triple for triple in itertools.permutations(indices, r=3))

    precedenceMatrix = utils.precedenceMatrix(data, n)

    # x_{i,j} is a binary variable since the full-ranking either places 
    # candidate i before candidate j, or it does not
    x_vars = {(i,j) : plp.LpVariable(cat=plp.LpBinary,name=f"{i},{j}") for (i,j) in pairs}

    # Set constraints

    # Any valid full-ranking has no ties, so we must have that 
    # x_{i,j} + x_{j,i} = 1
    for pair in pairs:
        i, j = pair
        model.addConstraint(plp.LpConstraint(
                            e=plp.LpAffineExpression(
                                [(x_vars[(i,j)], 1), (x_vars[(j,i)], 1)]),
                            sense=plp.LpConstraintEQ,
                            rhs=1))

    # Enforce transitivity: if i precedes j, and j precedes k, i must precede k
    for triple in triples:
        i,j,k = triple
        model.addConstraint(plp.LpConstraint(
                                        e=plp.LpAffineExpression(
                                            [(x_vars[(i,j)], 1), (x_vars[(j,k)], 1), (x_vars[(k,i)], 1)]),
                                        sense=plp.LpConstraintGE,
                                        rhs=1))
    
    # Define the objective function for Kemeny
    # If a list ranks j before i, then the contributed cost is the number 
    # of voters that ranked i before j, which is precedenceMatrix[i,j]
    # (and vice versa when i and j have swapped order)
    #
    # Note that this can be divided by N to get the
    # average Kendall-Tau distance from the optimal result to 
    # each of the lists, but the result will be twice as large 
    # because the same cost is incurred when considering 
    # x_(i,j) and x_(j,i) as when considering 
    # x_(j,i) and x_(i,j)
    kendall_dist = plp.lpSum(precedenceMatrix[i,j] *
                                x_vars[j,i] + 
                                    precedenceMatrix[j,i] * 
                                        x_vars[i,j] 
                                        for (i,j) in pairs)

    # Require Kemeny-distance to be minimized
    model.sense = plp.LpMinimize
    model.setObjective(kendall_dist)

    # msg = 0 suppresses log information
    model.solve(plp.PULP_CBC_CMD(msg=0))

    # Reconstruct ranking from solution.
    # Reconstruction is necessary for consistency with 
    # other algorithms, since they perform reconstruction

    # Dictionary to track how many candidates a given candidate precedes
    precedenceFreqency = {i:0 for i in indices}

    for var in model.variables():
        # Process the string name of variable
        name = var.name
        i = int(name.split(",")[0])

        # Update precedence frequency if i precedes a candidate j
        if var.varValue == 1:
            precedenceFreqency[i] += 1


    sigma = list()
    for candidate, frequency in precedenceFreqency.items():
        # In the final list, candidate must precede 
        # frequency candidates. One is subtracted since 
        # candidates do not precede themselves
        # 
        # For example, with 10 candidates,
        # the candidate that precedes everyone would precede 
        # 9 candidates since they don't precede themselves. 
        index = n - frequency - 1
        sigma.insert(index, candidate)

    time_elapsed = (time.process_time() - start_time) * 1000

    return ALGORITHM_NAME, utils.generalizedKendallTauDistance(data, sigma, n, N, s0), time_elapsed



     


    





