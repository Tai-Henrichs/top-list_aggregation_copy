import numpy as np

from os import listdir
from os.path import isfile, join

class Filter:

    def __init__(self, directory):
        self.directory = directory
        self.fnames = []
        self.topk = []

        self.ns = {}
        self.Ns = {}
        self.ths = {}
        self.ks = {}
        self.algos = {}

        self.readData(directory)

    
    def readData(self, dir):
        file_names = self.getfilenames(dir)
    
        for f in file_names:
            tokens = f.split("_")
    
            # if real dataset, ignore
            if tokens[0] != "mallows":
                continue
            # if topk instead of poisson, append topk
            if tokens[1] == "topk":
                self.topk.append(f)
                continue
    
            # process n
            try:
                curr_n = int(tokens[2][1:])
            except:
                print("Invalid n: ")
                print(tokens[2])
                return
            self.addToDict(self.ns, curr_n, f)
    
            # process N
            try:
                curr_N = int(tokens[3][1:])
            except:
                print("Invalid N: ")
                print(tokens[3])
                return
            self.addToDict(self.Ns, curr_N, f)
    
            # process th
            try:
                curr_th = float(tokens[4][2:])
            except:
                print("Invalid th: ")
                print(tokens[4])
                return
            self.addToDict(self.ths, curr_th, f)
    
            # process ks
            try:
                curr_k = float(tokens[5][1:-4]) / curr_n
                print("here")
            except:
                print("Invalid k: ")
                print(tokens[5])
                return
            self.addToDict(self.ks, curr_k, f)

            self.processAlgos(f) 

        print(self.ks)



    def getfilenames(self,dir):
        return [f for f in listdir(dir) if isfile(join(dir, f))]


    def addToDict(self, d, k, v):
        # generic adding to dict function
        if k in d:
            d[k] += [v]
        else:
            d[k] = [v]


    def processAlgos(self, fname):
        # to be used by filtering by algorithm
        with open(self.directory + fname) as f:
            for line in f:
                algo_label = line.strip().split(",")[0]
                algo_val = line.strip("\n") + ", " + fname[:-4] + "\n"
                self.addToDict(self.algos, algo_label, algo_val)


    def genericFilter(self, fnames):
        out_list = ["ALGORITHM, DISTANCE, TIME\n"]
        for fname in fnames:
            with open(self.directory + fname) as infile:
                next(infile)
                for line in infile:
                    out_list.append(line)
        return out_list


    def filter_by_k(self,k, outFile_name=None):
        if outFile_name == None:
            outFile_name = f'by_k{k}.csv'
        label = f'Filtering by k = {k}'
        data =  self.genericFilter(self.ks[k])
        self.writeToFile(data, outFile_name, label) 
        return data, outFile_name

    
    def filter_by_th(self,th, outFile_name=None):
        if outFile_name == None:
            outFile_name = f'by_th{th}.csv'
        label = f'Filtering by th = {th}'
        data =  self.genericFilter(self.ths[th])
        self.writeToFile(data, outFile_name, label) 
        return data, outFile_name

    
    def filter_by_n(self,n, outFile_name=None):
        if outFile_name == None:
            outFile_name = f'by_n{n}.csv'
        label = f'Filtering by n = {n}'
        data =  self.genericFilter(self.ns[n])
        self.writeToFile(data, outFile_name, label) 
        return data, outFile_name

    
    def filter_by_N(self,N, outFile_name=None):
        if outFile_name == None:
            outFile_name = f'by_N{N}.csv'
        label = f'Filtering by N = {N}'
        data =  self.genericFilter(self.Ns[N])
        self.writeToFile(data, outFile_name, label) 
        return data, outFile_name


    def filter_by_algo(self,alg_name, outFile_name=None):
        if outFile_name == None:
            outFile_name = f'by_{alg_name}.csv'
        label = f'Filter by Algorithm: {alg_name}'
        data = self.algos[alg_name]
        self.writeToFile(data, outFile_name, label)
        return data, outFile_name


    def get_ep_files(self):
        return self.topk


    def filter_by_param_and_algo(self, param_label, param_val, alg_name):
        if param_label == 'n':
            data = self.genericFilter(self.ns[param_val])
        elif param_label == 'N':
            data = self.genericFilter(self.Ns[param_val])
        elif param_label == 'th':
            data = self.genericFilter(self.ths[param_val])
        elif param_label == 'k':
            data = self.genericFilter(self.ks[param_val])
        else:
            print(f'Cannot filter by {param_label}. Try n, N, th, or k')
            return

        filtered_data =  [line for line in data if line.strip(" ").split(",")[0] == alg_name]

        file_name = f'by_{alg_name}_and_{param_label}{param_val}.csv'
        file_label = f'Filter by Algorithm: {alg_name} and by {param_label} = {param_val}'
        self.writeToFile(filtered_data, file_name, file_label)

        return filtered_data, file_name


    def writeToFile(self, data, filename, label=None):
        f = open(filename, "w")
        if label != None:
            f.write(f'{label}\n')
        for v in data:
            f.write(f'{v}')
        f.close()
