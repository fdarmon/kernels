# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:10:29 2018

@author: Alexandre Philbert
"""
import pandas as pd
import numpy as np

def extractdata(filename):
    """
        Extracts the  data given the path 
    """ 
    data  = pd.read_csv(filename,sep=',', header=None, engine='python').as_matrix()
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
