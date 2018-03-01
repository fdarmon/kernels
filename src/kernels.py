import numpy as np
class Kernel:
    """
    Each kernel must be similar to this class: 2 functions
        - gram_matrix(self,X)
        - predict(self,Xtrain,alphas,X)

    """

    def __init__(self, kernel_type = 'linear',deg = None, sigma = None):
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

    def gram_matrix(self,X):
        if self.type =='linear':
            return X @ X.T

        elif self.type =="polynomial":
            return (X @ X.T)**self.deg
        else:
            diff_matrix = X[None,:,:]-X[:,None,:] # shape n*n*d
            return(np.exp(-np.sum(diff_matrix**2,axis = 2)/(2*self.sigma**2)))

    def predict(self,X,Xtrain,alphas):
        """
        Given a training dataset Xtrain and the alphas trained on it,
        returns the prediction for the new dataset X
        """
        if self.type =='linear':
            return alphas.T @ Xtrain @ X.T

        elif self.type =="polynomial":
            return alphas.T @ (Xtrain @ X.T)**self.deg

        else:
            diff_matrix = Xtrain[:,None,:]-X[None,:,:] # shape n_train*n_test*d
            return(alphas.T @ np.exp(- np.sum(diff_matrix**2,axis = 2)/(2*self.sigma**2)))


def prediction_function(coef, data, kernel,x):
    n,p = np.shape(data)
    assert n == np.shape(coef)[0]
    tmp = np.array([coef[k]*kernel(data[k],x) for k in range(n)])
    return np.sum(tmp)


# Spectrum Kernel
def generate_uplets(alphabet,n,aux):
    if n==0:
        return aux
    else:
        tmp = []
        for a in (aux):
            for l in alphabet:
                tmp.append(l+a)

        return generate_uplets(alphabet,n-1,tmp)

def spectrum(sequence,k):
    """
    Computes the spectrum representation of the vector x
    Parameters
    ----------
    x : sequence
    k : length of substrings to consider
    Returns
    -------
    phix : spectrum representation of x. Array containing for each word
            of size k the number of occurences of the word in x
    """
    k_uplets = generate_uplets('ACTG',k,[''])
    occs = np.zeros(len(k_uplets))
    for i,s in enumerate(k_uplets):
        occs[i]=sequence.count(s)
    return occs
