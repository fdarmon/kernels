# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

import challengeFunctions as cf
import numpy as np

# EXTRACT AND CLEAN THE DATA
Xfile = "../data/Xtr0.csv"
Yfile = "../data/Ytr0.csv"
X,Y = cf.preprocessing(Xfile,Yfile)

#split into test set and validation set
X_train, Y_train, X_test, Y_test = cf.splitdata(X,Y)


print(X_train[0])
print(X[0])
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(Y_train))
print(np.shape(Y_test))
