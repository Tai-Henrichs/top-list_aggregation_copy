import sim
import itertools
import generate
import math
import utils

def test(data, params, testName, kemenyBound=None):
    """Execute tests on all implemented algorithms using 
       provided inputs. Performs various sanity checks 
       on the results.

        Parameters
        ----------
        data:  Counter or dict object 
                The keys  in this Counter are tuple top-lists and the 
                values are the mulitiplicities of each top-list. Both 
                the elements in the tuples and the values are ints.

        params: dict
              A python dictionary with that holds statics and info
              on the dataset stored on 'data'. Some of the keys are
              'n', 'N', 'k', and 'theta'. Refer to sim.py for full
              documentation

        testName: string
              A name for the test being executed.

        kemenyBound: boolean function that takes float as input
              Specifies a required range of values that all 
              algorithms' Kemeny-performance must satisfy.

    """
    line = utils.lineGenerator(10)
    print(f"\n{line}{testName}{line}")

    numCandidates = params['n']
    algorithms = sim.Simulation().funcDict
    testPassed = True

    allScores = dict()
    for name, func in algorithms.items():
        name, kemenyScore, _, sigma = func(data, params)
        allScores.setdefault(name, kemenyScore)

        if kemenyBound is not None and not kemenyBound(kemenyScore):
            print(f"{name} has an invalid Kemeny-score of {kemenyScore}")
            testPassed = False
        
        if len(sigma) != numCandidates:
            print(f"{name} produced an incomplete ranking: {sigma}")
            testPassed = False 
    
    optimalScore = allScores["OPTIMAL_SOLUTION"]
    for name, score in allScores.items():
        if optimalScore > score:
            print(
            f"{name}'s score of {score} beats the optimal score {optimalScore}!")

            testPassed = False

    print(f"\nTest Passed: {testPassed}")
    print(f"{line}{line}{line}")

# This file is for executing standardized tests 
# on all voting methods. None of the tests verify 
# the correctness of the output lists selected 
# by algorithms, only performing basic sanity 
# checks. For some tests, this is possible by 
# construction - any valid answer must produce 
# the same Kemeny-ranking for certain inputs.
#
# The exception to pure sanity checking 
# is the optimal algorithm, which is required to 
# perform no worse than any other alogorithm.
if __name__ == "__main__":
    data = dict()
    seed = 1
    theta = .01

    params = {'n': None, 'N': None, 'seed' : seed, 's0' : None, 'k' : None}

    # Test One
    name = "Simple All Possible Lists"

    params['n'] = 2
    params['N'] = 5
    params['k'] = params['n']

    data = { (0,1) : params['N']}
    kemenyBound = lambda num : num == 0 or num == 1

    test(data, params, name, kemenyBound)

    # Test Two
    name = "Simple Reversed Lists"

    params['n'] = 2
    params['N'] = 6 
    params['k'] = params['n']

    perList = params['N'] / 2

    data = { (0,1) : perList, (1,0) : perList}
    kemenyBound = lambda num : num == .5

    test(data, params, name, kemenyBound)

    # Test Three
    name = "All Possible Lists"

    params['n'] = 5
    candidates = tuple(i + 1 for i in range(params['n']))
    allPerms = tuple(perm for perm in itertools.permutations(candidates))

    perList = 3
    data = {perm:perList for perm in allPerms}
    params['N'] = perList * len(allPerms)
    params['k'] = params['n']

    kemenyBound = lambda num : num == params['n']

    test(data, params, name, kemenyBound)

    # Test Four 
    name = "Reversed Lists"

    params['n'] = 7
    params['N'] = 6 
    params['k'] = params['n']

    perList = params['N'] / 2

    candidates = tuple(i for i in range(params['n']))
    data = { candidates : perList, candidates[::-1] : perList}
    kemenyBound = lambda num : num == params['n']

    test(data, params, name)

    # Test Five
    name = "Mallows Top-K"

    params['n'] = 10
    params['N'] = params['n'] * 4
    params['k'] = math.ceil(params['n'] / 3)

    data = generate.MallowsSampleTopK(
                    params['N'],
                    params['n'],
                    params['k'],
                    theta=theta,
                    seed=seed
                    ).sample 
    
    test(data, params, name)

    # Test Six 
    name = "Mallows Poisson"

    params['n'] = 10
    params['N'] = params['n'] * 4

    # Set to one to guarantee that 
    # score-then-adjust will not 
    # attempt impossible 
    # permutations
    params['k'] = 1

    data = generate.MallowsSamplePoisson(
                    params['N'],
                    params['n'],
                    5,
                    theta=theta,
                    seed=seed
                    ).sample
    
    test(data, params, name)


    






