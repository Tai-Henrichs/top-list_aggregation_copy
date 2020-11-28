import os
import csv
import pandas as pd
import mallows_kendall as mk
import numpy as np

Sample top-k list from MM
numRankings = 2
theta = 0
k = 1
listLen = 5
# Without a provided consensus array, the function will utilize the
# identity  permutation by default
mk.sampling_top_k_rankings(m = numRankings, n = listLen, k = k, theta = theta)

#mk and numpy works!
perm1 = np.array([3,1,2,0,4])
print(mk.kendall_tau(perm1))
print(mk.max_dist(5)) #n(n-1)/2


# dump numpy array to CSV file
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
numpy.savetxt("foo.csv", a, delimiter=",")
#https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
#OR
pd.DataFrame(np_array).to_csv("path/to/file.csv", header=None, index=None)


#read CSV to numpy array
my_data = np.genfromtxt('my_file.csv', delimiter=',')

#convert numpy array to python list
p_list = ndarray.tolist(np_array)
