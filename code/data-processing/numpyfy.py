import numpy as np
import filter as F


#TODO: Update function names once new exps finish running
algorithms = ["FootRule+","RandomSort","Borda+","Score-Then-Borda+","Linear-Programming-Relaxation","Local-Search","Chanas","Copeland","QS RAND","QS DET","IS","OPTIMAL"]

# Number of algorithms: Q = 12 (poisson/toplists - excluding topk) 
Q = len(algorithms)

# params used to run exps
# TODO: update once new exps finish running
ns = [5,15,45]
Ns = [50,2000,5000]
ks_ratio = [.1,.5,.9]
ths = [.001,.01,.1]
epsilons = [.25, .5, .75, 1]

f = F.Filter("../results/")


def convert_to_npArray(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=1, usecols = (1,2))

def average(array):
    #print(f'Shape of array is: {np.shape(array)}')
    #print(f'Array is: {array}')

    m, _ = np.shape(array)
    return np.sum(array, axis=0) / m



# want to get the average of all algorithms for each value of n,N,th, or k
# will store in a 3x12x2 np.array (3 params, 12 algos, each with 2 data 
# points: time and accuracy)

def by(parameter):    
    out = np.empty((len(ns),Q,2))
    if parameter == 'n':
        l = ns
    elif parameter == 'N':
        l = Ns
    elif parameter == 'th':
        l = ths
    elif parameter == 'k':
        l = ks_ratio
    else:
        print("Invalid Parameter to fileter over!")

    for j in range(len(l)):
        algs = np.empty((Q,2))
        for i in range(Q):
            algo = algorithms[i]
            _, fname = f.filter_by_param_and_algo(parameter, l[j], algo)
            arr = convert_to_npArray(fname)
            algs[i] = average(arr)
            
        out[j] = algs
    return out


if __name__ == '__main__':
    print(by('n'))
    print(by('N'))
    print(by('th'))
    print(by('k'))
