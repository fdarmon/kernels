import numpy as np
class Kernel:
    """
    Each kernel must be similar to this class: 2 functions
        - gram_matrix(self,X)
        - predict(self,Xtrain,alphas,X)

    """

    def __init__(self, kernel_type = 'linear',deg = None, sigma = None, k_uplet = None):
        """
        Classical kernels
        kernel_type can be : ("linear" or "polynomial" or "gaussian")
        """
        if kernel_type in ['linear', 'polynomial','gaussian']:
            self.type = kernel_type
        else:
            print("Unknown kernel : {}".format(kernel_type))
            return
        if deg is None:
            self.deg = 2
        else:
            self.deg = deg

        if sigma is None:
            self.sigma = 1
        else:
            self.sigma = sigma

        if kernel_type =="spectrum":
            if k_uplet is None:
                self.k_uplet = generate_uplets('ACTG',3,[''])
            else:
                self.k_uplet = k_uplet

    def gram_matrix(self,X):
        if self.type =='linear':
            return X @ X.T

        elif self.type =="polynomial":
            return (X @ X.T)**self.deg

        elif self.type == "gaussian":
            diff_matrix = X[None,:,:]-X[:,None,:] # shape n*n*d
            return(np.exp(-np.sum(diff_matrix**2,axis = 2)/(2*self.sigma**2)))
        else :
            tmp = np.array([spectrum(X[i,0], 3, self.k_uplet) for i in range(np.shape(X)[0])])
            return tmp @ tmp.T


    def predict(self,X,Xtrain,alphas):
        """
        Given a training dataset Xtrain and the alphas trained on it,
        returns the prediction for the new dataset X
        """
        if self.type =='linear':
            return alphas.T @ Xtrain @ X.T

        elif self.type =="polynomial":
            return alphas.T @ (Xtrain @ X.T)**self.deg

        elif self.type == "spectrum":
            X_num = np.array([spectrum(X[i,0], 3, self.k_uplet) for i in range(np.shape(X)[0])])
            Xtrain_num = np.array([spectrum(Xtrain[i,0], 3, self.k_uplet) for i in range(np.shape(Xtrain)[0])])
            return alphas.T @ Xtrain_num @ X_num.T

        else:
            diff_matrix = Xtrain[:,None,:]-X[None,:,:] # shape n_train*n_test*d
            return(alphas.T @ np.exp(- np.sum(diff_matrix**2,axis = 2)/(2*self.sigma**2)))


def prediction_function(coef, data, kernel,x):
    n,p = np.shape(data)
    assert n == np.shape(coef)[0]
    tmp = np.array([coef[k]*kernel(data[k],x) for k in range(n)])
    return np.sum(tmp)




def spectrum(sequence,k,k_uplets):
    """
    Computes the spectrum kernel of the vector x
    Parameters
    ----------
    x : array containing the sequences in string format.
        Second dimension must be 1
    k : length of substrings to consider
    Returns
    -------
    phix : spectrum representation of x. Array containing for each word
            of size k the number of occurences of the word in x
    """
    #k_uplets = generate_uplets('ACTG',k,[''])
    occs = np.zeros(len(k_uplets))
    for i,s in enumerate(k_uplets):
        occs[i]=sequence.count(s)
    return occs

def spectrum_kernel(x1,x2,k=3):
    """
    Computes the spectrum kernel between x1 and x2
    """

    tmp1,tmp2 = x1.reshape(np.shape(x1)[0]),x2.reshape(np.shape(x2)[0])
    s1 , s2 = spectrum(tmp1,k),spectrum(tmp2,k)


    return np.dot(s1,s2.T)
