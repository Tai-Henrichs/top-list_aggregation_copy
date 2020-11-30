import sys
import numpy as np
import footule, randomsort, borda, scorethenborda, scorethenptas, scorethenadjust

from generate import MallowsSamplePoisson
from generate import MallowsSampleTopK

# import mallows_kendall as mk

"""
This is the class in charge of taking arguments from user and calling
appropriate functions.
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
            iv. the top-lists being at line n+3, are newline separated from each other, and the
            alternatives within each list are comman separated and ordered from left to right

    2. What algorithm(s) the user would like to use for top-list rank aggregation:
        a. 'FootRule+'
        b. 'RandomSort'
        c. 'Borda+'
        d. 'Score-Then-Borda+'
        e. 'Score-Then-PTAS' (might not implement)
        f. 'Score-Then-Adjust'

    -------------------------------

usage: python3 sim.py <s [OR] r>  <algo1,algo2=None, ...>  <[if s] n,N,theta,k>  <[if s] s0>  <[if r]: path/to/file.CSV>

Examples:
    python3 sim.py [Score-Then-Borda] s [10,100,0.5,3] [8,4,6,1,2,9,3,7,5,10] 

    python3 sim.py [RandomSort,Borda+,FootRule+] s [10,100,2,4] 

    python3 sim.py [Score-Then-Adjust] r data/soi/ED-00001-00000001.CSV 

"""

class Simulaton:

    def __init__(self):
        """
        Instance variables:
            'results' - a list of 3-tuples (algorithm, distance, time) that saves 
            performance info of each algorithm

            'algorithms' - list of function names (str)  to execute top list aggregation 
            on based on args

            'funcDict' - python dictionary mapping function name to function call from imports

            'data' - an object of type Counter in which input lists are stored
            with their repective frequencies. Each top-list (tuple) is the key and the frequency
            of such list is the value (int) 

            'params' - dataset information to keep track of like n, N, k, and s0
            Note:
                n, N, k are type (int) 
                s0 is a list of type (int)
                theta is type (float)


        """
        self.results = []

        self.algorithms = []

        self.funcDict = {
                "FootRule+": footrule.run, 
                "RandomSort": randomsort.run,
                "Borda+": borda.run, 
                "Score-Then-Borda+": scorethenborda.run, 
                "Score-Then-PTAS": scorethenptas.run, 
                "Score-Then-Adjust": scorethenadjust.run
                }

        self.data = None

        self.params = {
                'n': None,
                'N': None,
                'k': None,
                'theta': None,
                's0': None
                }



    def writeToFile(self):
        """
        Creates appropriate files for each algorithms (if doesn't exists already)
        and appends the comma separated string '<distance>, <time>' as a line
        
        """
        for c in self.results:
            label = c[0]
            if self.synth:
                label.append(F"{self.params['n']}, {self.params['N'], {self.params['theta']}}"
            f = open(label, "a+")  # append and read mode 
            f.write(F"{c[1]}, {c[2]} \n") 
            f.close()

            #TODO: pick up from here


    def genMallows(self, params):
        """
        This method returns a object of type Counter in which input lists are stored
        with their repective frequencies. Each top-list (tuple) is the key and the frequency
        of such list is the value (int)
        """
        return MallowsSamplePoisson(params).sample
        #return MallowsSampleTopK(params).sample         #manually switch between Poisson and TopK




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
        This method runs all the algorithms specifies in the list 'self.algorithms'
        for stored in self.data 

        After each iteration, results are updated by receiving a 3-tuple 
            (<algorithm name>, <kendall tau distance>, <time recorded>)
        from particular algorithm in 'alg'

        """
        for func in self.algorithms:
            if func not in self.funcDict:
                print("incorrect function name! {} was not found" .format(fun))
            alg = self.funcDict[func]

            #passes Counter object dataset as well as data specs
            self.results.append(alg(self.data, self.params)) 




    def __parseListArg__(self, s):
        """
        This is a helper for main
        """
        # remove "[" and "]"
        sp = s[1:-1]

        # split at every comma and save in list
        l = sp.split(",")

        #if parsing numerical list, convert to list of (int)
        if l[0].isnumeric():
            return [int(i) for i in l]

        #else leave as list of (str) case: parsing algorithms list
        return l


    def main(self, args):
        """
        This method takes in list of args and then calls genMallows() or parseCSV()
        to populate self.data depending on args. It uses the helper function parseListArg
        process comma separated arguments (instead of space separated) into a list 

        Finally, it calls handleFunc() and writeToFile()

        """
        arglen = len(args)

        self.algorithms = self.parseListArg(args[0])

        if args[1] == "r":
            self.data = self.parseCSV(args[2])

        elif args[1] == "s":
            params = self.parseListArg(args[2])

            # if 'k' is not specified
            if len(params) == 3:
                # default k is n/2 if user doesn't specify
                self.params['k'] = params[1] // 2

            # update params dict from parsed list
            self.params['n'] = params[0]
            self.params['N'] = params[1]
            self.params['theta'] = params[2]

            # extract user specified ground ranking if exists
            if arglen == 4:
                self.params['s0'] = s0self.parseListArg(args[3])

            # generate data
            self.data = self.genMallows(self.params)


        self.handleFunc()   # runs all algorithms
        self.writeToFile()  # writes results to files



if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("wrong usage. Please do: python3 sim.py <s [OR] r>  <algo1,algo2=None, ...>  <[if s] n,N,theta,k>  <[if s] s0=None>  <[if r]: path/to/file.CSV>")
    else:
        sim = Simulation()
        sim.main(sys.argv[1:])
        print(sim)

    print("\n Done!")
