from Classifier import SVM,LogisticRegression
from challengeFunctions import classification_accuracy,write_prediction_file
import kernels
import numpy as np
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make the SVM prediction for a given kernel and lambda')
    parser.add_argument('--kernel', type=int, default = 0,
                        help='Kernel folder (default 0)')
    parser.add_argument('--lambda_regularization', type = float,default = 0.1,
                        help='Regularization parameter for the training (default 0.1)')

    args = parser.parse_args()

    y_pred = [None for i in range(3)]
    kernels = [10,9,10]
    lambdas = [0.1,0.1,100]

    for dataset in range(3):
        y = np.loadtxt("./data/Ytr{}.csv".format(dataset),skiprows = 1, usecols = (1,),delimiter = ',')
        y = (y*2)-1 # 0/1 to -1/1
        n = y.shape[0]
        np.random.seed(2018)
        random_indexes  = np.random.permutation(np.arange(n))

        K = np.loadtxt("./computed_kernels/{}/train_{}.csv".format(kernels[dataset],dataset))
        #if dataset in [0,2]:
        K = K**2
            
        model = SVM()
        model.lamb = lambdas[dataset]

        model.train(K,y,verbose = True)
        K_test = np.loadtxt("./computed_kernels/{}/test_{}.csv".format(kernels[dataset],dataset))
        #if dataset in [0,2]:
        K_test = K_test**2
            
        y_pred[dataset] = (model.predict(K_test) > 0)

    write_prediction_file("submission.csv",y_pred)
