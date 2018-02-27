# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:10:29 2018

@author: Alexandre Philbert
"""
import pandas as pd
import numpy as np

def extractdata(filename, sep = ','):
    """
        Extracts the  data given the path
    """
    data  = pd.read_csv(filename,sep= sep, header=None, engine='python').as_matrix()
    return data


def splitdata(X,Y):
    n = np.shape(X)[0]
    idx = np.random.permutation(n)
    tridx = int(0.75*n)
    train_idx = idx[:tridx]
    test_idx = idx[tridx:]
    Xtrain, Y_train = X[train_idx,:], Y[train_idx]
    Xtest, Y_test = X[test_idx,:], Y[test_idx]
    return Xtrain, Y_train, Xtest, Y_test

def preprocessing(Xfile,Yfile):
    Xtmp = extractdata(Xfile)
    Y = extractdata(Yfile)
    Y =Y[1:,1]
    Y = Y.astype(int)
    Xtmp = np.array([list(Xtmp[k,0]) for k in range(np.shape(Xtmp)[0])])

    d = {'A':np.array([1,0,0,0]), 'C': np.array([0,1,0,0]), 'G':np.array([0,0,1,0]), 'T':np.array([0,0,0,1])}
    X = np.zeros((np.shape(Xtmp)[0], 4*np.shape(Xtmp)[1]))
    for k in range(np.shape(Xtmp)[1]):

        for r in range(np.shape(Xtmp)[0]):

            X[r,k*4:(k+1)*4] = d[Xtmp[r,k]]
    return X,Y

def classification_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true)/y_pred.shape[0]

def write_prediction_file(pred_file,preds):
    assert len(preds) == 3

    with open(pred_file,'w') as f:
        f.write("Id,Bound\n")
        for i,preds_i in enumerate(preds):
            for idx,p in enumerate(preds_i):
                f.write('{},{}\n'.format(idx+1000*i,int(p)))
