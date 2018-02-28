import quadprog
import numpy as np
import kernels
import cvxopt
from cvxopt import matrix
import time


class Classifier :
    def __init__(self):
        self.lamb = 1
        self.kernel_name = "linear"
        self.coef = 0
        self.kernel = self.setKernel(self.kernel_name)
        self.Xtrain = None
        self.predict_func= None
        self.solver = "cvxopt"
        self.bias = None

    def setKernel(self, name="manual", f= None, deg = None, sigma = None):
        legal_values = ["linear", "manual", "polynomial","gaussian"]
        assert name in legal_values
        self.kernel_name = name
        if name == "linear":
            f = lambda x,y : np.dot(np.ravel(x),np.ravel(y))
            self.kernel = f
        elif name == "polynomial":
            if deg == None:
                deg = 2
            f = lambda x,y : (np.dot(np.ravel(x),np.ravel(y)))**deg
            self.kernel = f
        elif name == "gaussian":
            if sigma == None:
                sigma = 1
            f = lambda x,y : np.exp(-np.dot(x-y,x-y)/(sigma**2))
            self.kernel = f
        else:
            assert f != None
            self.kernel = f

    def predict(self, X):
        n,d = np.shape(X)
        Y = np.array([self.predict_func(X[k]) + self.bias for k in range(n)])
        return Y

    def compute_K(self,X):

        n = X.shape[0]
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel(X[i],X[j])

        return K


class SVM(Classifier):
    def __init__(self):
        self.margin = 0
        super().__init__()

    def train(self, X, Y):
        # tcheck if the dimention match

        K = self.compute_K(X)

        n, d = X.shape
        assert(len(Y)==n)
        #define the QP
        if self.solver == "quadprog":

            G = 0.5*(K+K.T) + 10**(-8)*np.eye(n)
            a = Y
            C1 = -np.diag(Y)
            C2 = np.diag(Y)
            C = np.vstack([C1, C2]).T
            b1 = -np.ones(n)/(self.lamb*n*2)
            b2 = np.zeros(n)
            b = np.hstack([b1,b2])
            #solve the QP
            self.coef = quadprog.solve_qp(G, a, C, b)[0]

        else :

            P = .5 * (K + K.T)  # make sure P is symmetric
            args = [matrix(P), -matrix(Y)]
            C1 = np.diag(Y)
            C2 = -np.diag(Y)

            A = np.ones((1,Y.shape[0]))
            b = np.array([ 0.0 ])


            G = np.vstack([C1, C2])
            b1 = np.ones(n)/(self.lamb*n*2)
            b2 = np.zeros(n)
            h = np.hstack([b1,b2])
            args.extend([matrix(G), matrix(h)])
            args.extend([matrix(A), matrix(b)])

            sol = cvxopt.solvers.qp(*args)
            if 'optimal' not in sol['status']:
                print("no optimal solution")
            #solve the QP
            self.coef =  np.array(sol['x']).reshape((K.shape[1],))

        self.Xtrain = X
        self.predict_func = lambda x : kernels.prediction_function(self.coef, self.Xtrain, self.kernel,x)
        #compute the bias:
        tmp = Y*self.coef
        mask1 = tmp > 5*10**(-10)
        mask2 = tmp < (1/(self.lamb*n*2) - 5*10**(-10)*1/(self.lamb*n*2))
        mask = mask1*mask2
        nonSaturatedCoef = self.coef[mask]
        nonSaturatedy = Y[mask]
        nonSaturatedX = X[mask]
        print("Number of non saturated constraints : \n")
        print(nonSaturatedX.shape[0])
        tmp = np.array([1/nonSaturatedy[k] - self.predict_func(nonSaturatedX[k]) for k in range(nonSaturatedX.shape[0]) ])
        self.bias = np.mean(tmp)


class LogisticRegression(Classifier):
    def __init__(self):
        super().__init__()

    def logistic(self,x):
        return(1/(1+np.exp(np.clip(-x,-20,20))))

    def train(self,X,Y):
        tol = 1e-8
        K = self.compute_K(X)
        n ,d = X.shape
        assert(len(Y)==n)

        self.coef = np.zeros((n,))

        tic = time.time()
        while(True):

            m = K @ self.coef
            sqrt_w = np.sqrt(self.logistic(m)*self.logistic(-m))
            z = m + Y/self.logistic(-Y * m)

            mat_sqrt_W = np.diag(sqrt_w)
            mat_sqrt_iW = np.diag(1/sqrt_w)

            A = (mat_sqrt_W @ K @ mat_sqrt_W + n * self.lamb * np.eye(n)) @ mat_sqrt_iW
            b = mat_sqrt_W @ z
            new_coef = np.linalg.solve(A,b)
            print(self.logistic(m))
            if np.all(np.abs(self.logistic(m)-self.logistic(K @ new_coef)) < tol):
                break
            else:
                self.coef = new_coef

        print("Training done in {}s".format(time.time()-tic))
        self.Xtrain = X
        self.predict_func = lambda x : kernels.prediction_function(self.coef, self.Xtrain, self.kernel,x)
