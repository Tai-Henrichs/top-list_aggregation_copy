import numpy as np
import sim

from utils import *
from generate import MallowsSamplePoisson

simulation = sim.Simulation()
data2 = simulation.parseCSV("../data/soi/test.csv")

n, N = 12, 5345
print(unrankedAlternatives(data2, n, N))
print(scores(data2, n, N))
print(avgRanks(data2, n, N))
