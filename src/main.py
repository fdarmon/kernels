# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

import challengeFunctions as cf
import numpy as np

# EXTRACT AND CLEAN THE DATA
Xtmp = cf.extractdata("Xtr0.csv")
Y = cf.extractdata("Ytr0.csv")
Y =Y[1:,1]
Y = Y.astype(int)
Xtmp = np.array([list(Xtmp[k,0]) for k in range(np.shape(Xtmp)[0])])

d = {'A':np.array([1,0,0,0]), 'C': np.array([0,1,0,0]), 'T':np.array([0,0,1,0]), 'G':np.array([0,0,0,1])}
X = np.zeros((np.shape(Xtmp)[0], 4*np.shape(Xtmp)[1]))

for k in range(np.shape(Xtmp)[1]):
    
    for r in range(np.shape(Xtmp)[0]):
        
        X[r,k*4:(k+1)*4] = d[Xtmp[r,k]]
        

#split into test set and validation set 
X_train, Y_train, X_test, Y_test = cf.splitdata(X,Y)


print(X_train[0])
print(X[0])
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(Y_train))
print(np.shape(Y_test))