#import quadprog
import numpy as np
import kernels
import cvxopt
from cvxopt import matrix
import time


class Classifier :
    """
    Abstract class from which every classifier must inherit
    A classifier must have a train function that given a training gram matrix
    and corresponding label trains the model into 'coef'
    """
    def __init__(self,lamb = 1):
        """


        """
        self.lamb = lamb

        self.coef = None


    def predict(self, test_matrix):
        """
        Returns the prediction of the dataset

        params:
            X : numpy array of shape (n_train,n_test) array of K(x_i,x_j) for x_i
            in the training set and x_j in the set to predict

        """
        if self.coef is None:
            print("Error, the classifier has not been trained")
        else:
            assert self.coef.shape[0] == test_matrix.shape[0]
            return self.coef.T @ test_matrix




class SVM(Classifier):
    """
    Class that implements kernel SVM
    """
    def __init__(self):
        super().__init__()
        self.solver = "cvxopt"

        self.bias = None

    def predict(self,X):
        return super().predict(X)# + self.bias

    def train(self, K, Y, verbose = False):
        """
        Function to train the SVM and put the result in self.coef

        params :
            K : gram matrix of the training data
            Y : labels
        """
        n = K.shape[0]
        assert(len(Y)==n)
        #define the QP
        if self.solver == "quadprog":
            return
            """
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
            """

        else :
            if not verbose:
                cvxopt.solvers.options['show_progress'] = False
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


        #compute the bias: skipped for now
        
        tmp = Y*self.coef
        mask1 = tmp > 5*10**(-10)
        mask2 = tmp < (1/(self.lamb*n*2) - 5*10**(-10)*1/(self.lamb*n*2))
        mask = mask1*mask2
        nonSaturatedCoef = self.coef[mask]
        nonSaturatedy = Y[mask]
        nonSaturatedX = K[:,mask]
        #print("Number of non saturated constraints : \n")
        #print(nonSaturatedX.shape[0])
        tmp = 1/nonSaturatedy - self.predict(nonSaturatedX)
        self.bias = np.mean(tmp)
        


class LogisticRegression(Classifier):
    """
    Class that implements Kernel Logistic Regression with an Iteratively
    reweighted least square.
    """
    def __init__(self):
        super().__init__(kernel)

    def logistic(self,x):
        return(1 / ( 1 + np.exp(np.clip( -x , -20 , 20) ) ) )

    def train(self,K,Y):
        """
        Function to train the logreg and put the result in self.coef

        params :
            K : gram matrix of the training data
            Y : labels
        """

        tol = 1e-8
        n = X.shape[0]
        assert(len(Y)==n)

        self.coef = np.zeros((n,))

        tic = time.time()
        last_loss = np.inf
        while(True):
            m = K @ self.coef
            sqrt_w = np.sqrt( self.logistic(m) * self.logistic(-m))
            z = m + Y / self.logistic(-Y * m)

            mat_sqrt_W = np.diag(sqrt_w)
            mat_sqrt_iW = np.diag(1/sqrt_w)

            A = ((mat_sqrt_W @ K @ mat_sqrt_W) + n * self.lamb * np.eye(n)) @ mat_sqrt_iW
            b = mat_sqrt_W @ z
            new_coef = np.linalg.solve(A,b)

            loss_function =  1/n*np.sum( - np.log(self.logistic(Y*m)))+self.lamb/2*self.coef.T@ K @self.coef
            print("loss : {}".format(loss_function))
            if loss_function > last_loss + 1e-4:
                print("Error, increasing loss")
                break

            if np.abs(loss_function-last_loss) < tol:
                break
            else:
                self.coef = new_coef
                last_loss = loss_function


        print("Training done in {}s".format(time.time()-tic))
