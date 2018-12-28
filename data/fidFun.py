
# coding: utf-8

# In[64]:

import numpy as np


# In[65]:

def sqrtMat(mat, flag=False):
    """
    sqrtMat(2-d numpy array, Bool) --> 2-d numpy array
    
    Takes as input a 2-d numpy array, assuming it is
    hermitian and positive semidefinite upto numerical
    accuracy, returns the square root of the matrix.
    
    
    Arguments:
        mat  :  A 2-d numpy array
        flag :  Optional Boolean argument, if True
                returns only the trace of the square root.
    Returns
        sqrtMat : Positivie Semi-definite square 
                  root of the matrix
    """
    M1 = (mat + mat.T.conj())/2.
    (val, vec) = np.linalg.eigh(M1)
    
    eps = min(val)
    
    if eps < 0. and abs(eps) < 1e-10:
        val = val + abs(eps)
    
    sqrtVal = np.sqrt(val)
    
    if flag:
        return np.sum(sqrtVal)
    
    sqrtMat = np.dot(sqrtVal*vec, vec.conj().T)
    return sqrtMat


# In[66]:

def fidelity(M1, M2):
    """
    fidelity(2-d numpy array, 2-d numpy array) --> float
    
    Takes as input two positive semi-definite matrices
    and computes the fidelity between them. See eq. (9.53)
    of Quantum Computation and Quantum Information
    10th Anniversary Edition
    
    Arguments:
        M1 : A Semi-definite matrix
        M2 : A Semi-definite matrix
    
    Returns:
        Fidelity between M1 and M2. 
    """
    sqrtM1 = sqrtMat(M1)
    
    rho = np.dot(np.dot(sqrtM1, M2), sqrtM1)
    sigma = sqrtMat(rho, True)
    
    return sigma

