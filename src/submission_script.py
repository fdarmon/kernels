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
    for dataset in range(3):
        y = np.loadtxt("./data/Ytr{}.csv".format(dataset),skiprows = 1, usecols = (1,),delimiter = ',')
        y = (y*2)-1 # 0/1 to -1/1
        k_fold = 5
        n = y.shape[0]
        np.random.seed(2018)
        random_indexes  = np.random.permutation(np.arange(n))

        K = np.loadtxt("./computed_kernels/{}/train_{}.csv".format(args.kernel,dataset))
        K = K/np.mean(K)

        model = SVM()
        model.lamb = args.lambda_regularization

        model.train(K,y)
        K_test = np.loadtxt("./computed_kernels/{}/test_{}.csv".format(args.kernel,dataset))
        K_test = K_test/np.mean(K_test)
        y_pred[dataset] = (model.predict(K_test) > 0)

    write_prediction_file("submission.csv",y_pred)
