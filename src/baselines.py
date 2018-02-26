from challengeFunctions import preprocessing, splitdata, classification_accuracy, write_prediction_file
import numpy as np


def train_logreg(X,y):
    n = X.shape[0]
    p = X.shape[1]

    omega=np.zeros(p)

    while True:
        eta=1 / (1 + np.exp(np.clip(-X @ omega,-10,10)))

        D_eta=np.diag(eta*(1-eta))

        hessian = np.linalg.inv( X.T @ D_eta @ X )
        add = hessian @ X.T @ (y - eta)
        if np.sum(add >=1e-8) == 0:
            break
        omega = omega + add

    return omega
y_preds = []
for i in range(3):
    X = np.genfromtxt('../data/Xtr{}_mat50.csv'.format(i))
    y = np.genfromtxt('../data/Ytr{}.csv'.format(i),skip_header = 1,delimiter = ',')[:,1]
    omega = train_logreg(X,y)

    y_pred = np.ones(X.shape[0])
    y_pred[X @ omega < 0] = 0
    print("Training accuracy : {}".format(classification_accuracy(y_pred, y)))

    X_test = np.genfromtxt('../data/Xte{}_mat50.csv'.format(i))
    y_pred = np.ones(X_test.shape[0])
    y_pred[X_test @ omega < 0] = 0
    y_preds.append(y_pred)
write_prediction_file('pred.csv',y_preds)
