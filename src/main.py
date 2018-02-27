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
offset = np.ones((X_train.shape[0],1))
X_tr = np.hstack([X_train,offset])
offset = np.ones((X_test.shape[0],1))
X_te = np.hstack([X_test,offset])
svm = Classifier.SVM()
svm.lamb = 0.001
svm.setKernel("gaussian")
svm.train(X_tr,Y_tr)
Y_p = svm.predict(X_te)
Y_p = Y_p > 0
Y_p = 2*Y_p - 1
res = np.sum(Y_p == Y_te)/X_test.shape[0]

print(res)

# print(X_train[0])
# print(X[0])
# print(np.shape(X_train))
# print(np.shape(X_test))
# print(np.shape(Y_train))
# print(np.shape(Y_test))
