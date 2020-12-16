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

# Varying epsilon with three sets of other params, one low, one medium, and one high 
# by low we mean a smaller value for n, N, and k, and larger for th (smaller and high consensus dataset)
# by high we mean a larger value for n, N, and k, and smaller for th (harder and low consensus dataset)
# by medium we mean a dataset of average or mean difficulty between high and low

# Total epsilon datasets = 8*3 = 16

# n = 10, N = 50, th = 0.1, k = 2 
master += [f'python3 sim.py [Opt,Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] s [10,50,0.1,2] c {SEED}'
            for ep in epsilons
            ]

# n = 30, N = 500, th = 0.01, k = 15 
master += [f'python3 sim.py [Opt,Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] s [30,500,0.01,15] c {SEED}'
            for ep in epsilons
            ]

# n = 50, N = 5000, th = 0.001, k = 45 
master += [f'python3 sim.py [Opt,Score-Then-Adjust,Score-Then-Adjust-Relaxed,{ep}] s [50,5000,0.001,45] c {SEED}'
            for ep in epsilons
            ]


#--------------------------------------


""" # REAL WORLD DATA

# Total # real world datasets: 38, i.e. len(fnames) = 38

PATH = '../data/soi/'

fnames = [f for f in listdir(PATH) if isfile(join(PATH, f))]

master += [f'python sim.py [FootRule+,RandomSort,Borda+,Score-Then-Borda+,Relaxed-Linear-Program,Local-Search,Chanas,Copeland,QS-Rand,QS-Det,IS,Opt] r {PATH}{fname} c {SEED}'
            for fname in fnames
            ] """

#--------------------------------------


# RUN EXPERIMENTS:

for experiment in master:
    sys(experiment)
