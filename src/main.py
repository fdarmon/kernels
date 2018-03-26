# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:13:43 2018

@author: Alexandre Philbert
"""

from Classifier import SVM,LogisticRegression
from challengeFunctions import classification_accuracy,write_prediction_file
import kernels
import numpy as np
import os



k_fold = 5
np.random.seed(2018)

lambdas = [1e-6,0.1,100]
kernels = [7,8,9,10,11]

nb_kernel = len(kernels)
dirname = 0

while(os.path.exists("./res/{}".format(dirname))):
    dirname = dirname + 1
os.mkdir("./res/{}".format(dirname))

print("Created directory {} for writing the results".format(dirname))
with open("./res/{}/config.txt".format(dirname),'w') as f:
    f.write("Res {}\n{} fold Crossvalidation\n kernels squared : {}\n lambdas : {}\n".format(dirname,k_fold,kernels,lambdas))

for dataset in range(3):

    y = np.loadtxt("./data/Ytr{}.csv".format(dataset),skiprows = 1, usecols = (1,),delimiter = ',')
    y = (y*2)-1 # 0/1 to -1/1
    n = y.shape[0]
    random_indexes  = np.random.permutation(np.arange(n))

    print("Cross validation for dataset {}".format(dataset))

    train_acc = np.zeros((len(kernels),len(lambdas)))
    val_acc = np.zeros((len(kernels),len(lambdas)))

    for idx,kernel in enumerate(kernels):
        
        K = np.loadtxt("./computed_kernels/{}/train_{}.csv".format(kernel,dataset))
        print("\t kernel {}".format(kernel))
        K = K/np.sqrt(np.outer(np.diag(K),np.diag(K)))
        K = (K)**2
            
        #K = K/np.mean(K)
        #K = K/np.diag(K)
        #K = K/np.sqrt(np.outer(np.diag(K),np.diag(K)))
        for idl,lamb in enumerate(lambdas):
            print("\t\t Lambda : {}".format(lamb))
            vals = np.zeros(k_fold)
            trains = np.zeros(k_fold)
            for j in range(k_fold):
                train_indexes = np.concatenate([random_indexes[0:j*n//k_fold],random_indexes[(j+1)*n//k_fold:]])
                val_indexes = random_indexes[j*n//k_fold:(j+1)*n//k_fold]

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
                    print("Cannot train SVM with degree {}".format(deg))
            print("\t \t Validation score : {}".format(np.mean(vals)))
            val_acc[idx,idl] = np.mean(vals)
            train_acc[idx,idl] = np.mean(trains)

    np.savetxt("res/{}/val_acc_dataset_{}.csv".format(dirname,dataset),val_acc)
    np.savetxt("res/{}/train_acc_dataset_{}.csv".format(dirname,dataset),train_acc)
