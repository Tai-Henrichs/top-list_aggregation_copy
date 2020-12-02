import sys
#import footule, randomsort, borda, scorethenborda, scorethenptas, scorethenadjust

from collections import Counter
from generate import MallowsSamplePoisson
#from generate import MallowsSampleTopK


"""
This is the class in charge of taking arguments from user and calling
appropriate functions.
------------------------------

The different choices that we provide are:
    1. What dataset does the user want to use
        a. 's' (synthetic) for which we require the following arguments:

                'n' : int 
                    the size of the full rankings lists (also the total
                    number of alternatives)
                'N' : int 
                    the total number of partial lists to be generated
                'theta' : float 
                    the parameter that is passed to the Mallows Model that 
                    dictates the probability distribution (consensus)
                'k' [optional] : int 
                    the distribution median (i.e. average candidates ranked)
                    across all voters)
                's0' [optional] : int[]
                    the ground list

        b. 'r' (real), only requires a well formatted /path/to/file.CSV. By well
        formatted, we require that:

            i.  The first row has the number 'n' of alternatives

            ii. The following 'n' rows only have two columns: the first is the i'th 
                candiate (int) and the second is a (str) describes the candiate

            iii.The n+2'nd row must have as its first column the total numbers
                of voters N (size of the input)
            
            iv. The top-lists being at line n+3, are newline separated from each other, and the
                alternatives within each list are comman separated and ordered from left to right

    2. What algorithm(s) the user would like to use for top-list rank aggregation:
        a. 'FootRule+'
        b. 'RandomSort'
        c. 'Borda+'
        d. 'Score-Then-Borda+'
        e. 'Score-Then-PTAS' (might not implement)
        f. 'Score-Then-Adjust'

    -------------------------------

Usage: python3 sim.py <s [OR] r>  <algo1,algo2=None, ...>  <[if s] n,N,theta,k>  <[if s] s0>  <[if r]: path/to/file.CSV>

Examples:

    python3 sim.py [Score-Then-Borda] s [10,100,0.5,3] [8,4,6,1,2,9,3,7,5,10] 

    python3 sim.py [RandomSort,Borda+,FootRule+] s [10,100,2,4] 

    python3 sim.py [Score-Then-Adjust] r data/soi/ED-00001-00000001.CSV 

"""

# documentation for the follwing code is a little informal at the moment but will iteratively be
# improved as the program develops

class Simulation:

    def __init__(self):
        """
        Instance variables:

            'results' : list of 3-tuples  (str, float, float)
                        saves performance info of each algorithm (algorithm, distance, time)

            'funcDict' : dict
                        maps function name to function call from imports

            'data' :    an object of type Counter in which input lists are stored
                        with their repective frequencies. Each top-list (tuple) is the key and 
                        the frequency of such list is the value (int) 

            'params' : dataset information to keep track of like n, N, k, and s0

        """
        self.results = []

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
                'label': "results/",
                'n': None,
                'N': None,
                'k': None,
                'theta': None,
                's0': None
                }



    def __str__(self):
        """
        Formats dataset statistics and output of simulation for each algorithm
        """
        out = "DATASET INFO:\n"
        for key, val in self.params.items():
            out += f'{key}: {val}\n'

        out += "\nEXPERIMENTS:\n"
        for alg, acc, time in self.results:
            out += f'{alg} ran in {time} seconds with a score of {acc}\n'
        
        return out



    def writeToFile(self):
        """
        Creates appropriate files for each algorithms based on the label params
        (if doesn't exists already) and appends the comma separated string 
        '<distance>, <time>' as a line.

        """
        for c in self.results:
            fname = f'{self.params["label"]}_{c[0]}.txt'
            f = open(fname, "a") 
            f.write(f'{c[1]}, {c[2]}\n') 
            f.close()

        # TODO: might wanna modify this at some point for different datasets with different
        # values of k and s0. Should not matter much for now


    def genMallows(self, params):
        """
        This method returns a object of type Counter in which input lists are stored
        with their repective frequencies. Each top-list (tuple) is the key and the frequency
        of such list is the value (int)

        Note: this separate method was created in order to swtich between poisson and topk
              in the future
        """

        return MallowsSamplePoisson(params).sample




    def parseCSV(self, path):
        """
        This method takes a path (string) to a file then processes it contents to create
        a Counter object in which preference lists will be stored as keys and their
        respective occurances as values.

        We also make sure to update N and n. Note: there is no ground truth s0 nor dispersion
        variable theta.
        """

        c = Counter()

        with open(path) as f:
            # save n
            self.params['n'] = int(next(f))

            # skip n next lines (candidate info)
            for _ in range(self.params['n']):
                next(f)
            
            #save N
            self.params['N'] = next(f).split(",")[0]

            for line in f:
                #convert line to list of ints and ignore newline character
                toplist = [int(i) for i in line.split(',')[:-1]]
                #preferences ordering does not include its count
                toptuple = tuple(toplist[1:])

                #assign count to ordering and put it in Counter object
                c[toptuple] = toplist[0]

        return c


    def handleFunc(self, algorithms):
        """
        This method runs all the algorithms specifies in the list 'self.algorithms'
        for stored in self.data 

        After each iteration, results are updated by receiving a 3-tuple 
            (<algorithm name>, <kendall tau distance>, <time recorded>)
        from particular algorithm in 'alg'

        """
        for func in algorithms:
            if func not in self.funcDict:
                print(f'incorrect function name! {func} was not found')
            alg = self.funcDict[func]

            #passes Counter object dataset as well as data specs
            # TODO: assumption that all algorithms return (<algorithm name>, <kendall tau distance>, <time recorded>)
            self.results.append(alg(self.data, self.params)) 



    def parseListArg(self, s):
        """
        This is a helper for main that serves to process list-like command line args
        Takes string and appropriately transforms into list of strings or ints.
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

        if args[1] == "r":
            # setting label according to file name if real data
            self.params['label'] +=  args[2].split("/")[-1]
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
                self.params['s0'] = self.parseListArg(args[3])

            # generate data
            self.data = self.genMallows(self.params)

            # setting label according to Mallows distribution, n, N, and theta
            self.params['label'] += f'poisson_n={self.params["n"]}_N={self.params["N"]}_th={self.params["theta"]}'

        else:
            print("wrong usage! second argument should be 'r' or 's'")
            return


        #run all algorithms
        self.handleFunc(self.parseListArg(args[0])) 

        #write all results to files
        self.writeToFile()



if __name__ == '__main__':
    if not (3 <= len(sys.argv) <=  5):
        print("wrong usage. Please do: python3 sim.py <s [OR] r>  <algo1,algo2=None, ...>  <[if s] n,N,theta,k>  <[if s] s0=None>  <[if r]: path/to/file.CSV>")
    else:
        sim = Simulation()
        sim.main(sys.argv[1:])
        print(sim)

    print("\nDone!")
