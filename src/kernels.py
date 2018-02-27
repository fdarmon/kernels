import numpy as np

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
    Computes the spectrum kernel of the vector x
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
