import numpy as np
import pandas as pd
import time
from numba import jit
from multiprocessing.pool import ThreadPool
import sys
import argparse
import os
data_dir = "./data/"

def load_data(n,t):
    """
    n: index of the dataset to be loaded
    t:type of data , 'train' or 'test'
    """
    if(t=='test'):
        path = data_dir+'Xte'+str(n) +'.csv'
        data = pd.read_csv(path,header=-1)
        X = data.as_matrix()
        X = X.reshape(np.shape(X)[0]) # Sequences in String format
        return X
    if(t=='train'):
        xpath = data_dir+'Xtr' +str(n) +'.csv'
        ypath = data_dir+'Ytr' +str(n) +'.csv'
        data = pd.read_csv(xpath,header=-1)
        labels = pd.read_csv(data_dir+'Ytr0.csv')
        X = data.as_matrix()
        X = X.reshape(np.shape(X)[0]) # Sequences in String format
        Y = labels['Bound'].as_matrix()
        return X,Y
    else:
        print('unknown type of dataset')


@jit(nopython = True, nogil = True)
def kernel_func(inputs,K = 2, lambda_param = 0.9):
    ### ========================================================================================
    ### B_k computation
    ### ========================================================================================
    x1,x2 = inputs
    B_k_1 = np.ones((len(x1),len(x2)))
    #[[1 for i in range(len(x2))] for j in range(len(x1))]# # B_k_1[i,j] = B_{k-1}(x1[0:i+1],x2[0:j+1])
    for k in range(K-1):

        B_k =np.zeros((len(x1),len(x2)))# [[0 for i in range(len(x2))] for j in range(len(x1))]
        #) # B_k[i,j] = B_k(x1[0:i+1],x2[0:j+1])

        B_k[k,k] = lambda_param**(2*(k+1))
        for idx in range(k+1):
            if x1[idx] != x2[idx]:
                B_k[k,k] = 0
                break

        for i in range(k+1,len(x1)):
            res = 0
            for l in range(1,k+1):
                if x2[l] == x1[i]:
                    res = B_k_1[i-1,l-1]*lambda_param**(k-l+1)


            B_k[i,k] = lambda_param*B_k[i-1,k] + res

        for j in range(k+1,len(x2)):
            res = 0
            for l in range(1,k+1):
                if x1[l] == x2[j]:
                    res = B_k_1[l-1,j-1]*lambda_param**(k-l+1)
            B_k[k,j] = lambda_param*B_k[k,j-1] + res

        for i in range(k+1,len(x1)):
            for j in range(k+1,len(x2)):
                if x1[i] == x2[j]:
                    tmp = lambda_param**2 * B_k_1[i-1,j-1]
                else:
                    tmp = 0

                B_k[i,j] = lambda_param*(B_k[i,j-1] + B_k[i-1,j]) - lambda_param**2* B_k[i-1,j-1] + tmp

        B_k_1 = B_k
    ### ========================================================================================
    ### K_k computation
    ### ========================================================================================
    K_k = np.zeros((len(x1),len(x2)))#[[0 for i in range(len(x2))] for j in range(len(x1))] # K_k[i,j] = K_k(x1[0:i+1],x2[0:j+1])
    k = K - 1 # to have the same notation as above where k is in range(K)
    K_k[k,k] = lambda_param**(2*(k+1))
    for i in range(k+1):
        if x1[i] != x2[i]:
            K_k[k,k] = 0
    for i in range(k,len(x1)):
        if i > k:
            res = 0
            for l in range(1,k+1):
                if x2[l] == x1[i]:
                    res = res + B_k_1[i-1,l-1]

            K_k[i][k] = K_k[i-1][k] + res*lambda_param**2

        for j in range(k+1,len(x2)):
            res = 0
            for l in range(1,i+1):
                if x1[l] == x2[j]:
                    res = res + B_k_1[l-1,j-1]

            K_k[i,j] = K_k[i,j-1] + res*lambda_param**2

    return(K_k[-1,-1])

def convert_to_list(str_in,dico):
    res = []
    for l in str_in:
        res.append(dico[l])
    return np.array(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Substring kernel : computes training and testing kernel matrices.')
    parser.add_argument('--K', type=int, default = 2,
                        help='Length of substrings (default 2)')
    parser.add_argument('--lambda_param', type = float,default = 0.9,
                        help='Lambda parameter of the substring kernel (default 0.9)')
    parser.add_argument("--dataset",type = int, default = None, help = "Dataset to process, by default all")
    parser.add_argument("--nb_threads",type = int, default = 4, help = "Number of threads for paralel computation (default 4)")

    args = parser.parse_args()
    dico = {'A' : 0, 'C' : 1,'G':2,'T':3}

    if args.dataset is None:
        datasets = [0,1,2]
        dirname = 0
        while(os.path.exists("./computed_kernels/{}".format(dirname))):
            dirname = dirname + 1
        os.mkdir("./computed_kernels/{}".format(dirname))
        print("Created directory {} for writing the results".format(dirname))
        with open("./computed_kernels/{}/config.txt".format(dirname),'w') as f:
            f.write("kernel  = substring\nK = {}\nLambda = {}".format(args.K,args.lambda_param))

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
        l = []
        l_t = []

        for i in range(n):
            l.append(convert_to_list(X[i],dico))

        for i in range(n_t):
            l_t.append(convert_to_list(X_t[i],dico))

        raveled_l = []
        raveled_lt = []

        res = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                raveled_l.append((l[i],l[j]))

        for i in range(n):
            for j in range(n_t):
                raveled_lt.append((l[i],l_t[j]))

        func  = lambda x : kernel_func(x,lambda_param = args.lambda_param,K = args.K)



        tic = time.time()
        with ThreadPool(args.nb_threads) as p:
            res = p.map(func,raveled_l)

        mat_res = np.zeros((n,n))
        cpt = 0
        for i in range(n):
            for j in range(i,n):
                mat_res[i,j] = res[cpt]
                mat_res[j,i] = res[cpt]
                cpt = cpt+1

        print("Finished computing training matrix of dataset {} in {}s".format(dataset_nb,time.time()-tic))



        filename = 'computed_kernels/{}/train_{}.csv'.format(dirname,dataset_nb)
        np.savetxt(filename,mat_res)

        tic = time.time()
        with ThreadPool(args.nb_threads) as p:
            res = p.map(func,raveled_lt)

        mat_res = np.zeros((n,n_t))
        cpt = 0
        for i in range(n):
            for j in range(n_t):
                mat_res[i,j] = res[cpt]
                cpt = cpt+1

        print("Finished computing testing matrix of dataset {} in {}s".format(dataset_nb,time.time()-tic))

        filename = 'computed_kernels/{}/test_{}.csv'.format(dirname,dataset_nb)
        np.savetxt(filename,mat_res)
