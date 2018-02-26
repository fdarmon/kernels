import quadprog
import numpy as np
import kernels


class SVM :
    def __init__(self):
        self.lamb = 1
        self.kernel_name = "linear"
        self.coef = 0
        self.margin = 0
        self.kernel = self.setKernel(self.kernel_name)
        self.Xtrain = None
        self.predict_func= None

    def setKernel(self, name, f= None ):
        legal_values = ["linear", "manual"]
        assert name in legal_values
        self.kernel_name = name
        if name == "linear":
            f = lambda x,y : np.dot(x,y)
            self.kernel = f
        else:
            assert test != None
            self.kernel = f

    def train(self, X, Y):
        # tcheck if the dimention match
        shapex = np.shape(X)
        shapy = np.shape(Y)

        assert shapex[0] == shapy[0]

        n, d = shapex[0], shapex[1]
        #define the Kernel matrix
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel(X[i],X[j])

        #define the QP
        G = 0.5*(K+K.T) + 10**(-10)*np.eye(n)
        a = Y
        C1 = np.diag(-Y)
        C2 = np.diag(Y)
        C = np.hstack([C1, C2])
        b1 = -np.ones(n)/(self.lamb*n)
        b2 = np.zeros(n)
        b = np.hstack([b1,b2])

        #solve the QP
        self.coef = quadprog.solve_qp(G, a, C, b)[0]
        self.Xtrain = X
        self.predict_func = lambda x : kernels.prediction_function(self.coef, self.Xtrain, self.kernel,x)


    def predict(self, X):
        n,d = np.shape(X)
        Y = np.array([self.predict_func(X[k]) for k in range(n)])
        return Y
