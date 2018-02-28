# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

import challengeFunctions as cf
import Classifier
import kernels
import numpy as np

# EXTRACT AND CLEAN THE DATA
Xfile = "../data/Xtr0.csv"
Yfile = "../data/Ytr0.csv"
X,Y = cf.preprocessing(Xfile,Yfile)

#split into test set and validation set
X_train, Y_train, X_test, Y_test = cf.splitdata(X,Y)

Y_tr = 2*(Y_train - 1/2)
Y_te = 2*(Y_test - 1/2)
svm = Classifier.SVM()
svm.lamb = 1
kernelName = ["gaussian", "linear", "polynomial"]
for name in kernelName:
    print(name,"\n")
    svm.setKernel(name)
    svm.train(X_train,Y_tr)
    Y_p = svm.predict(X_train)
    Y_p = 2*Y_p - 1
    res = np.sum(Y_p == Y_tr)/X_train.shape[0]
    print("training accuracy : ",res,"\n")
    Y_p = svm.predict(X_test)
    Y_p = 2*Y_p - 1
    res = np.sum(Y_p == Y_te)/X_test.shape[0]
    print("test accuracy : ",res,"\n")

# print(X_train[0])
# print(X[0])
# print(np.shape(X_train))
# print(np.shape(X_test))
# print(np.shape(Y_train))
# print(np.shape(Y_test))
