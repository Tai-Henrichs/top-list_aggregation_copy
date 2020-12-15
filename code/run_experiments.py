from os import system as sys
from os import listdir
from os.path import isfile, join

import numpy as np


master = []
SEED = 0

#-------------------------------------

# SYNTHETIC DATA

ns = [5,15,30,45,60]
Ns = [50,200,800,2000,5000]
ks_ratio = [0.1,0.3,0.5,0.7,0.9]
ths = [0.001,0.01,0.1,1]
epsilons = [i for i in np.arange(0.1,0.91,0.05)]


# Combination of varying n, N, k, and ths for all algorithms except for all Score-Then-Adjust variants
# total combinations  datasets 5*5*5*4 = 500 

master += [f'python3 sim.py [Chanas,Quick-Sort-Rand,Quick-Sort-Det,Insertion-Sort] s [{n},{N},{th},{k*n}] {SEED}'
            for k in ks_ratio
            for n in ns
            for N in Ns
            for th in ths
            ]

# Varying epsilon with two sets of other params, one low and one high
# by low we mean a smaller value for n, N, and k, and larger for th (smaller and high consensus dataset)
# by high we mean a larger value for n, N, and k, and smaller for th (harder and low consensus dataset)
#
# Total epsilon datasets = 18*2 = 36
#
#LOW = 1 
#HIGH = 3 
#
## We don't run optimal here as it has already been ran for the same datasets in the combinations above
#
##n=15, N=200, th=0.1, k=5
#master += [f'python3 sim.py [Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep},Optimal] s [{ns[LOW]},{Ns[LOW]},{ths[HIGH-1]},5] {SEED}'
#        for ep in epsilons
#        ]
#
##n=45, N=2000, th=0.001, k=30
#master += [f'python3 sim.py [Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep},Optimal] s [{ns[HIGH]},{Ns[HIGH]},{ths[LOW-1]},30] {SEED}'
#        for ep in epsilons
#        ]
#
#--------------------------------------
#
#
## REAL WORLD DATA
#
## Total # real world datasets: 34, i.e. len(fnames) = 34
#
#PATH = '../data/soi/'
#
#fnames = [f for f in listdir(PATH) if isfile(join(PATH, f))]
#
#master += [f'python3 sim.py [FootRule+,Borda+,Score-Then-Borda+,RandomSort,Local-Search,Relaxed-Linear-Program,Copeland,Optimal] r {PATH}{fname} {SEED}'
#            for fname in fnames
#            ]
#
## no need for optimal here, already computed
#master += [f'python3 sim.py [Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] r {PATH}{fname} {SEED}'
#            for ep in epsilons
#            for fname in fnames
#            ]
#
#
#--------------------------------------


# RUN EXPERIMENTS:

for experiment in master:
    sys(experiment)
