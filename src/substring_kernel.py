import numpy as np
import pandas as pd
import time
from numba import jit
from multiprocessing.pool import ThreadPool
data_dir = "../data/"

def load_data(n,t):
    """
    n: index of the dataset to be loaded
    t:type of data , 'train' or 'test'
    """
    if(t=='test'):
        path = data_dir+'Xte'+str(n) +'.csv'
        data = pd.read_csv(path,header=-1)
        X = data.as_matrix()
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
            for l in range(k+1):
                if x2[l] == x1[i]:
                    res = B_k_1[i-1,l]*lambda_param**(k-l+1)


            B_k[i,k] = lambda_param*B_k[i-1,k] + res

        for j in range(k+1,len(x2)):
            res = 0
            for l in range(k+1):
                if x1[l] == x2[j]:
                    res = B_k_1[l,j-1]*lambda_param**(k-l+1)
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

    for i in range(k,len(x1)):
        res = 0
        for l in range(k):
            if x2[l+1] == x1[i]:
                res = res + B_k_1[i-1,l]

        K_k[i][k] = K_k[i-1][k] + res*lambda_param**2

        for j in range(k+1,len(x2)):
            for l in range(i):
                if x1[l+1] == x2[j]:
                    res = res + B_k_1[l,j-1]

            K_k[i,j] = K_k[i,j-1] + res*lambda_param**2

    return(K_k[-1,-1])

def convert_to_list(str_in,dico):
    res = []
    for l in str_in:
        res.append(dico[l])
    return np.array(res)

if __name__ == '__main__':
    X,Y=load_data(0,'train')
    dico = {'A' : 0, 'C' : 1,'G':2,'T':3}

    n = X.shape[0]
    l = []

    for i in range(n):
        l.append(convert_to_list(X[i],dico))
    raveled_l = []
    tic = time.time()
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            raveled_l.append((l[i],l[j]))

    with ThreadPool(50) as p:
        res = p.map(kernel_func,raveled_l)
    mat_res = np.zeros(n,n)
    cpt = 0
    for i in range(n):
        for j in range(n):
            mat_res[i,j] = res[cpt]
            cpt = cpt+1
    print(time.time()-tic)
    np.savetxt(mat_res,"../res.csv")
