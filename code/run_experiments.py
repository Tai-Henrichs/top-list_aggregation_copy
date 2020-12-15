from os import system as sys
from os import listdir
from os.path import isfile, join

import numpy as np


master = []
SEED = 0

#-------------------------------------

# SYNTHETIC DATA

ns = [10,30,50]
Ns = [50,500,5000]
ks_ratio = [.1,.5,.9]
ths = [.001,.01,.1]
epsilons = [.25, .5, .75, 1]


# Combination of varying n, N, k, and ths for all algorithms except for all Score-Then-Adjust variants
# total combinations  datasets 5*5*5*4 = 500 

master += [f'python sim.py [FootRule+,RandomSort,Borda+,Score-Then-Borda+,Relaxed-Linear-Program,Local-Search,Chanas,Copeland,QS-Rand,QS-Det,IS,Opt] s [{n},{N},{th},{k*n}] c {SEED}'
            for k in ks_ratio
            for n in ns
            for N in Ns
            for th in ths
            ]

# Varying epsilon with two sets of other params, one low and one high
# by low we mean a smaller value for n, N, and k, and larger for th (smaller and high consensus dataset)
# by high we mean a larger value for n, N, and k, and smaller for th (harder and low consensus dataset)

# Total epsilon datasets = 18*2 = 36

LOW = 1 
HIGH = 3 

# We don't run optimal here as it has already been ran for the same datasets in the combinations above

#n=15, N=200, th=0.1, k=n*0.3
master += [f'python3 sim.py [Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] s [{ns[LOW]},{Ns[LOW]},{ths[HIGH-1]},{ks_ratio[LOW]*ns[LOW]:.0f}] c {SEED}'
            for ep in epsilons
            ]

#n=45, N=2000, th=0.001, k=n*0.7
master += [f'python3 sim.py [Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] s [{ns[HIGH]},{Ns[HIGH]},{ths[LOW-1]},{ks_ratio[HIGH]*ns[HIGH]:.0f}] c {SEED}'
            for ep in epsilons
            ]

#--------------------------------------


# REAL WORLD DATA

# Total # real world datasets: 38, i.e. len(fnames) = 38

PATH = '../data/soi/'

fnames = [f for f in listdir(PATH) if isfile(join(PATH, f))]

master += [f'python3 sim.py [FootRule+,RandomSort,Borda+,Score-Then-Borda+,Relaxed-Linear-Program,Local-Search,Chanas,Copeland,Quick-Sort-Rand,Quick-Sort-Det,Insertion-Sort,Opt] r {PATH}{fname} c {SEED}'
            for fname in fnames
            ]

#--------------------------------------


# RUN EXPERIMENTS:

for experiment in master:
    sys(experiment)