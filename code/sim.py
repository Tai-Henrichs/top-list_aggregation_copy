import sys
import numpy as np
import footule, randomsort, borda, scorethenborda, scorethenptas, scorethenadjust

from generate import MallowsSamplePoisson
from generate import MallowsSampleTopK

# import mallows_kendall as mk

"""
This is the class in charge of taking arguments from user and calling
appropriate funcitons.
------------------------------

The different choices that we provide are:
    1. What dataset does the user want to use
        a. 's' (synthetic) for which we require the following arguments:
            i. 'n' (int)- the size of the full rankings lists (also the total
            number of alternatives)
            ii. 'N' (int)- the total number of partial lists to be generated
            iii. 'theta' (float)- the parameter that is passed to the Mallows
            Model that dictates the probability distribution (consensus)
            iv. 'k' (int) [optional]- the distribution median (i.e. average candidates ranked)
            across all voters)
            v. 's0' [optional] (int[])- which is the ground list

        b. 'r' (real), only requires a well formatted /path/to/file.CSV. By well
        formatted, we require that:
            i. The first row has the number 'n' of alternatives
            ii. The following 'n' rows only have two columns: the first is the
            i'th candiate (int) and the second is a (str) describes the candiate
            iii. The n+2'nd row must have as its first column the total number
            of voters N (size of the input)

    2. What algorithm(s) the user would like to use for top-list rank aggregation:
        a. 'FootRule+'
        b. 'RandomSort'
        c. 'Borda+'
        d. 'Score-Then-Borda+'
        e. 'Score-Then-PTAS' (might not implement)
        f. 'Score-Then-Adjust'

    -------------------------------

Usage: python3  sim.py  <s [OR] r>  <[if s] n,N,theta,k>  <[if s] s0=None>
       <[if r]: path/to/file.CSV>  <algo1, algo2=None, ...>

Examples:
    python3 sim.py s [10,100,0.5,3] [8,4,6,1,2,9,3,7,5,10] [Score-Then-Borda]

    python3 sim.py s [10,100,2,4] [RandomSort, Borda+, FootRule+]

    python3 sim.py r data/soi/ED-00001-00000001.CSV [Score-Then-Adjust]

"""

class Simulaton:

    def __init__(self):
        self.results = list()
        self.algorithms = list()

        self.funcDict = {"FootRule+": footrule.run, "RandomSort": randomsort.run,\
        "Borda+": borda.run, "Score-Then-Borda+": scorethenborda.run, \
        "Score-Then-PTAS": scorethenptas.run, "Score-Then-Adjust": scorethenadjust.run}

        self.data = None
        self.params = None



    def writeToFile(self):
        pass



    def genMallows(self, params):
        """
        This method returns a object of type Counter in which input lists are stored
        with their repective frequencies. Each list (tuple) is the key and the frequency
        of such list is the value (int)
        """
        return MallowsSamplePoisson(params).sample
        #return MallowsSampleTopK(params).sample




    def parseCSV(self, path):
        #don't forget to asssing self.data and self.params

        #maybe use:

        # import csv
        # with open(dest_file,'r') as dest_f:
        #     data_iter = csv.reader(dest_f, delimiter = delimiter, quotechar = '"')
        #     data = [data for data in data_iter]
        # data_array = np.asarray(data, dtype = <whatever options>)





    def handleFunc(self):
        """
        This method accesses runs all the algorithms specifies in the list 'self.algorithms'
        for each sampled dataset in self.data (numpy array).

        After each iteration, self.results is updated by running self.updateResults()
        """
        for func in self.algorithms:
            if func not in self.funcDict:
                print("incorrect function name! {} was not found" .format(fun))
            alg = self.funcDict[func]
            self.updateResults(alg(self.data))




    def updateResults(self, output= {}):
        pass





    def main(self, args):
        """
        Thi method takes in list of args and then calls genMallows() or parseCSV()
        to populate self.data. It then calls handleFunc() and writeToFile()

        Params:
        --------------
        args: a lit of args as defined by <s [OR] r>  <[if s] n,N,theta,k>
              <[if s] s0=None>  <[if r]: path/to/file.CSV>  <algo1, algo2=None, ...>

        """
        arglen = len(args)

        if args[0] == "r":
            self.data = self.parseCSV(args[1])

        else:
            #remove "[" and "]"
            params = args[1]
            params = params[1:-1]
            params = params.split(",")
            if len(params) == 3:
                #default k is n/2 if user doesn't specify
                params.append(params[1] // 2)

            #we don't need to worry about ground ranking, not specified by user
            if arglen == 3:
                self.data = self.genMallows(params)

            #extract ground ranking and make it part of params for data generation
            else arglen == 4:
                #remove "[" and "]"
                s0 = args[2]
                s0 = s0[1:-1]
                s0 = s0.split(",")
                params.append(s0)
                self.data = self.genMallows(params)

            #update self.params for later use by other methods
            self.params = params


        if args[2] == None:
            print("wrong usage! arg[2] is null")
            return
        #remove "[" and "]"
        algos = args[2]
        algos = algos[1:-1]
        #update self.algorithms fir later use by other methods
        self.algorithms = algos.split(",")


        self.handleFunc()   #runs all algorithms
        self.writeToFile()  #writes results to file'

        ## TODO: Method is broken! Fix!!!




if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("wrong usage. Please do: python3  sim.py  <s [OR] r>  <[if s] n,N,\
        theta,k>  <[if s] s0=None>  <[if r]: path/to/file.CSV>  <algo1,algo2=None\
        , ...>")
    else:
        sim = Simulation()
        sim.main(sys.argv[1:])
        sim.writeToFile()
        print(sim)

    print("\n Done!")
