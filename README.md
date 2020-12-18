# CS 357 Final Project Repository: A Study of Top-Lists Aggregation Methods

### What is this?
A final project for Algorithmic Game Theory at Williams College where we, Ammar Eltigani and Tai Henrichs, implemented and compared a bunch of approximation algorithms to the top-list aggregation problem. Read our paper in this repo!
#### Requirements: 
python3, numpy, scipy, pulp, and Gurobi. We recommend running the following commands
	
	python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
	pip install pulp
And obtaining an academic license for the Gurobi LIP solver on you machine at https://www.gurobi.com/

The different choices that we provide are:

    1. What dataset does the user want to use
        a. 's' (synthetic) for which we require the following arguments:
                'n' : int 
                    the size of the full rankings lists (also the total
                    number of alternatives)
                'N' : int 
                    the total number of partial lists to be generated
                'theta' : float 
                    the parameter that is passed to the Mallows Model that 
                    dictates the probability distribution (consensus)
                'k' [optional] : int 
                    the distribution median (i.e. average candidates ranked)
                    across all voters)
                's0' [optional] : int[]
                    the ground list, which should order the n candidates 1,2,...,n
				'ep' (epsilon) If Score-Then-Adjust or Score-Then-Adjust Relaxed are part of the algorithms list, this parameter is required. Note: it gets passed with the algorithms list, not the parameters list

        b. 'r' (real), only requires a well formatted /path/to/file.CSV. By well
        formatted, we require that:

            i.  The first row has the number 'n' of alternatives

            ii. The following 'n' rows only have two columns: the first is the i'th 
                candiate (int) and the second is a (str) describes the candiate

            iii.The n+2'nd row must have as its first column the total numbers
                of voters N (size of the input)
            
            iv. The top-lists being at line n+3, are newline separated from each other, and the
                alternatives within each list are comman separated and ordered from left to right

    2. What algorithm(s) the user would like to use for top-list rank aggregation:
        a. 'FootRule+'
        b. 'RandomSort'
        c. 'Borda+'
        d. 'Score-Then-Borda+'
        e. 'Score-Then-Adjust'
        f. 'Local-Search'
        g. 'Relaxed-Linear-Program'
        h. 'Score-Then-Adjust-Relaxed'
        i. 'Copeland'
        j. 'Chanas"
        k. 'QS-Rand'
        l. 'QS-Det'
        m. 'IS'
        n. 'Opt'
        
	3. Whether to run combinations of algorithms 'c' or not 'nc'. Combinations entails in running Chanas and Local-Search as a postprocessing step to every other algorithm.
	
    4. An optinal seed argument. If provided, all random number generation will utilize the given seed. By default, 
        random number generation will utilize the system's internal clock. 

    -------------------------------

#### Usage: 

		python3 sim.py [algo1,algo2,...,epsilon] s [n,N,theta,k] s0[OPTIONAL] c<OR>nc seed
		python3 sim.py [algo,algo2,...] r FILEPATH c

#### Examples:

	python3 sim.py [Opt] s [10,100,0.5,3] [8,4,6,1,2,9,3,7,5,10] nc 0

    python3 sim.py [Chanas,RandomSort,Borda+,FootRule+] s [10,100,2,4] c

    python3 sim.py [FootRule+] r ../data/soi/ED-00001-00000001.csv nc

    python3 sim.py [Score-Then-Adjust,0.2,0.4,0.5,Score-Then-Borda+] s [5,50,0.5] nc 25

