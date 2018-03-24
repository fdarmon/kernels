from substring_kernel import load_data
from multiprocessing import Pool
import argparse
import os
import numpy as np
import time


class Retrieval_tree:
    def __init__(self,letter):
        """
        Letter : letter to access the node in the tree
        childrens : list of childrens ([] if leaf)
        value : number of time accessed
        """
        self.letter = letter
        self.childrens = []
        self.value = 0

    def init_prefix(self,string,p):
        """
        Init a prefix tree from the list of length-p prefixes of string
        """
        for i in range(len(string)-p):
            self.add_str(string[i:i+p])


    def add_str(self,x):
        """
        Add a string to the tree
        """
        self.value = self.value + 1

        if len(x) > 0:
            found  = False
            for i,c in enumerate(self.childrens):
                if c.letter == x[0]:
                    found = True
                    self.childrens[i].add_str(x[1:])
                    break
            if not found:
                new_node = Retrieval_tree(x[0])
                new_node.add_str(x[1:])
                self.childrens.append(new_node)

    def disp(self, level = 0):
        if len(self.childrens) == 0:
            ret = "    "*level + self.letter + "->" + str(self.value)
        else:
            ret = "    "*level + self.letter
        print(ret)
        for c in self.childrens:
            c.disp(level = level+1)

    def get_value(self,key):
        if len(key) > 0:
            res = 0
            for i,c in enumerate(self.childrens):
                if c.letter == key[0]:
                    res = c.get_value(key[1:])
                    break

            return(res)

        else:
            return self.value

    def kernel(self, x2, p):
        res = 0
        for i in range(len(x2)-p):
            res = res + self.get_value(x2[i:i+p])

        return res

def get_row(x1,x2_l,p):
    """
    returns the row of the kernel matrix K_p(x1,x2_0) ... K_p(x1,x2_n)
    """
    root = Retrieval_tree("root")
    root.init_prefix(x1,p)
    res = np.zeros(len(x2_l))
    for i,x2 in enumerate(x2_l):
        res[i] = root.kernel(x2,p)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectrum kernel : computes training and testing kernel matrices.')
    parser.add_argument('--p', type=int, default = 2,
                        help='Length of substrings (default 2)')
    parser.add_argument("--dataset",type = int, default = None, help = "Dataset to process, by default all")
    parser.add_argument("--nb_threads",type = int, default = 4, help = "Number of threads for paralel computation (default 4)")

    args = parser.parse_args()
    if args.dataset is None:
        datasets = [0,1,2]
        dirname = 0
        while(os.path.exists("./computed_kernels/{}".format(dirname))):
            dirname = dirname + 1
        os.mkdir("./computed_kernels/{}".format(dirname))
        print("Created directory {} for writing the results".format(dirname))
        with open("./computed_kernels/{}/config.txt".format(dirname),'w') as f:
            f.write("kernel  = spectrum\np = {}\n".format(args.p))

    else:
        if not os.path.exists("./computed_kernels/tmp"):
            os.mkdir("./computed_kernels/tmp")
        dirname = 'tmp'
        datasets = [args.dataset]

    for dataset_nb in datasets:
        X,Y=load_data(dataset_nb,'train')
        X_t = load_data(dataset_nb,'test')

        n = X.shape[0]
        n_t = X_t.shape[0]

        with open("data/Xtr{}.csv".format(dataset_nb),'r') as f:
            raw_l = f.readlines()

        l = [s[:-2] for s in raw_l]

        with open("data/Xte{}.csv".format(dataset_nb),'r') as f:
            raw_l_t = f.readlines()

        l_t = [s[:-2] for s in raw_l_t]

        def func(x):
            return(get_row(x,l,args.p))

        tic = time.time()
        with Pool(args.nb_threads) as p:
            res = p.map(func,l)

        mat_res = np.array(res)

        print("Finished computing training matrix of dataset {} in {}s".format(dataset_nb,time.time()-tic))
        print(mat_res.shape)


        filename = 'computed_kernels/{}/train_{}.csv'.format(dirname,dataset_nb)
        np.savetxt(filename,mat_res)

        tic = time.time()
        with Pool(args.nb_threads) as p:
            res = p.map(func,l_t)

        mat_res = np.array(res).T

        print("Finished computing testing matrix of dataset {} in {}s".format(dataset_nb,time.time()-tic))

        filename = 'computed_kernels/{}/test_{}.csv'.format(dirname,dataset_nb)
        np.savetxt(filename,mat_res)
