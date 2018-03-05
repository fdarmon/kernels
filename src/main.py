# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

import challengeFunctions as cf
import Classifier
import kernels
import numpy as np

Xfile = "../data/Xtr0.csv"
Yfile = "../data/Ytr0.csv"
X,Y = cf.extract(Xfile,Yfile)
Xtrain, Y_train, Xtest, Y_test = cf.splitdata(X,Y)
# test the different lambda for the spectrum kernel 
train_acc = []
test_acc = []
lambs = [1e-8,1e-5,1e-3,1e-2,1,10,100]
for lamb in lambs:
    svm = Classifier.SVM("spectrum")
    svm.lamb = lamb

    Y_train_svm = 2*(Y_train - 1/2)
    Y_test_svm = 2*(Y_test - 1/2)
    svm.train(Xtrain, Y_train_svm)
    y_pred = svm.predict(Xtrain)
    y_pred = y_pred > 0
    train_acc.append(cf.classification_accuracy(Y_train, y_pred))
    y_pred = svm.predict(Xtest)
    y_pred = y_pred > 0
    test_acc.append(cf.classification_accuracy(Y_test, y_pred))

plt.semilogx(lambs,train_acc)
plt.semilogx(lambs,test_acc)
plt.legend(["Training",'Test'])
plt.show()
