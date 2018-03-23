# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

from Classifier import SVM,LogisticRegression
from challengeFunctions import classification_accuracy,write_prediction_file
import kernels
import numpy as np

dataset = 0
y = np.loadtxt("./data/Ytr{}.csv".format(dataset),skiprows = 1, usecols = (1,),delimiter = ',')
y = (y*2)-1 # 0/1 to -1/1
k_fold = 5
n = y.shape[0]
np.random.seed(2018)
random_indexes  = np.random.permutation(np.arange(n))

lambdas = [0.0001,0.001,0.01,0.1,1,10,100,1000]
kernels = [0,1,2,3]
nb_kernel = len(kernels)
train_acc = np.zeros((nb_kernel,len(lambdas)))
val_acc = np.zeros((nb_kernel,len(lambdas)))

for idk,kernel in enumerate(kernels):
    K = np.loadtxt("./computed_kernels/{}/train_{}.csv".format(kernel,dataset))
    K = K/np.mean(K)
    for i,lamb in enumerate(lambdas):
        print("Lambda = {}".format(lamb))
        vals = np.zeros(k_fold)
        trains = np.zeros(k_fold)
        for j in range(k_fold):
            train_indexes = np.concatenate([random_indexes[0:j*n//k_fold],random_indexes[(j+1)*n//k_fold:]])
            val_indexes = random_indexes[i*n//k_fold:(j+1)*n//k_fold]

            model = SVM()
            model.lamb = lamb
            try:
                model.train(K[train_indexes[:,None],train_indexes[None,:]],y[train_indexes])

                y_train = model.predict(K[train_indexes[:,None],train_indexes[None,:]])
                y_train = (y_train > 0 )*2 -1

                y_pred = model.predict(K[train_indexes[:,None],val_indexes[None,:]])
                y_pred = (y_pred > 0 )*2 -1

                vals[j] = classification_accuracy(y_pred,y[val_indexes])
                trains[j] = classification_accuracy(y_train,y[train_indexes])
            except:
                print("Cannot train SVM with lambda = {} for kernel {}".format(lamb,kernel))
        val_acc[idk,i] = np.mean(vals)
        train_acc[idk,i] = np.mean(trains)

np.savetxt("res/val_acc_dataset_{}.csv".format(dataset),val_acc)
np.savetxt("res/train_acc_dataset_{}.csv".format(dataset),train_acc)
with open("res/config.txt",'w') as f:
    f.write("Lambdas : {}\n Kernels : {}".format(lambdas,kernels))
