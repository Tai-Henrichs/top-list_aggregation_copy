import numpy as np
import sim

from utils import *
from generate import MallowsSamplePoisson

simulation = sim.Simulation()
data2 = simulation.parseCSV("../data/soi/ED-00001-00000001.csv")

n, N = 12, 5345
print(unrankedAlternatives(data2, n, N))
print(scores(data2, n, N))
print(avgRanks(data2, n, N))

#12
#1,Cathal Boland F.G.  
#2,Clare Daly S.P.  
#3,Mick Davis S.F.  
#4,Jim Glennon F.F.  
#5,Ciaran Goulding Non-P 
#6,Michael Kennedy F.F.  
#7,Nora Owen F.G.  
#8,Eamonn Quinn Non-P 
#9,Sean Ryan Lab 
#10,Trevor Sargent G.P.  
#11,David Henry Walshe C.C. Csp 
#12,G.V. Wright F.F.  
#5345,43942,19299

#800,12,6,4
#680,4,6,12
#506,6,12,4
#486,12,4,6
#429,6,4,12
#367,4,12,6
#343,10
#278,2
#251,9
#194,9,10,2
#177,4
#172,2,9,10
#171,12
#168,9,2,10
#163,6
#159,7,1,9,10
