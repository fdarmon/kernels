import numpy as np

def prediction_function(coef, data, kernel,x):
    n,p = np.shape(data)
    assert n == np.shape(coef)[0]
    tmp = np.array([coef[k]*kernel(data[k],x) for k in range(n)])
    return np.sum(tmp)
