import numpy as np
import search
import sim 
import run_experiments
import collections
import pandas as pd

sim = sim.Simulation()
# base algorithms 
# deque since indexed access not required
algorithms = collections.deque(sim.funcDict.keys())

# functionality not required/workable for algorithms with epsilon parameters
algorithms.remove("Score-Then-Adjust")
algorithms.remove("Score-Then-Adjust-Relaxed")

algorithmsBaseCopy = tuple(algorithms)

# Keep separate set of combination algorithms for later re-use
comboAlgos = set()

# add combination algorithms
postProcessAlgos = ["Chanas", "Local-Search"]
for postProcessAlgo in postProcessAlgos:
    # Iterate over algorithmsBaseCopy
    for initialAlgorithm in algorithmsBaseCopy:
        if not initialAlgorithm == postProcessAlgo and not initialAlgorithm == "Opt":
            algoName = f"{initialAlgorithm}_{postProcessAlgo}"
            algorithms.append(algoName)
            comboAlgos.add(algoName)

f = search.Search("../Synthetic-Results/")
algorithms = list(algorithms)

def convert_to_npArray(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=1, usecols = (1,2))

def average(array):
    m, _ = np.shape(array)
    return np.sum(array, axis=0) / m

def listByParam(parameter):
    if parameter == 'n':
        l = run_experiments.ns
    elif parameter == 'N':
        l = run_experiments.Ns
    elif parameter == 'th':
        l = run_experiments.ths
    elif parameter == 'k':
        l = run_experiments.ks_ratio
    else:
        l = None 
    
    return l

def by(parameter): 
    numAlgorithms = len(algorithms)   
    
    l = listByParam(parameter)

    # want to get the average of all algorithms for each value of n,N,th, or k
    # will store in a len(l)xnumAlgorithmsx2 np.array (params, algos, each with 2 data 
    # points: time and accuracy)
    out = np.empty((len(l),numAlgorithms,2))

    for j in range(len(l)):
        algs = np.empty((numAlgorithms,2))
        for i in range(len(algorithms)):
            _, fname = f.filter_by_param_and_algo(parameter, l[j], algorithms[i])
            arr = convert_to_npArray(fname)
            algs[i] = average(arr)
            
        out[j] = algs
    return out

def tidydf(parameter):
    l = listByParam(parameter)

    info = by(parameter)

    headers = ["Algorithms", parameter, "Average Kendall-Tau Distance", "Time (CPU Seconds)"]
    output = list()

    # Fill in time and distance data
    for i in range(len(info)):
        table = info[i]
        for j in range(len(table)):
            distance, time = table[j]
            rowToAdd = [algorithms[j], l[i], distance, time]
            output.append(rowToAdd)
            
    return pd.DataFrame(output, columns=headers)

if __name__ == '__main__':
    print(f'By n: {run_experiments.ns}\n {tidydf("n")}\n')
    print(f'By N: {run_experiments.Ns}\n {tidydf("N")}\n')
    print(f'By th: {run_experiments.ths}\n {tidydf("th")}\n')
    print(f'By k: {run_experiments.ks_ratio}\n {tidydf("k")}\n')
