
# coding: utf-8

# In[3]:

import numpy as np


# In[4]:

#This file contains all the functions needed for the projected
#gradient descent standalone verson
#List of function
#isValidBloch(vec)      :   Checks whether bloch vector is valid
#PSDProjBloch(vec)      :   Projects bloch vector onto sphere
#arrNoChange(pts)       :   Checks if last few points are different
#dualGap(vec,vec)       :   Computes surrogate duality gap for qubits


# In[5]:

#Takes a bloch, returns whether it is Positive Semi-definite or not
#Tested: YES
def isValidBloch(vec):
    """
    isValidBloch(vec, tol) ->  True/False

    Checks if the vector has length less than 1.
    
    Arguments:
        mt              : bloch vector as numpy array
    
    Returns:
        pos : True if Bloch vector is valid; False otherwise.

    Uses:
        numpy as np
    """
    
    return np.dot(vec,vec) <= 1.


# In[6]:

def PSDProjBloch(vec):
    """
    PSDProjBloch(1-d numpy array) --> 1-d numpy array

    Takes a numpy array as input, projects it onto
    a sphere.

    Arguments:
        vec :   A numpy array representing bloch vector

    Returns
        Projection of vector onto unit ball
    
    Uses:
        numpy as np, probProj
    """
    nrm = np.linalg.norm(vec)
    if nrm > 1.:
        return vec/nrm
    return vec

# In[8]:

def arrNoChange(arr, k = 4, tol = 1e-10):
    """
    arrNoChange(list, optional int, optional float)
    
    Arguments:
        arr  : A list of numbers
        
        k[optional]    : An integer representing how many previous values to compare
        tol[optional]  : A small constant to check closeness of floating point numbers
    
    Returns
        True if the last k values in the array are close to the final value
        False if array is smaller than k or values are not close.
    
    Uses:
        numpy as np
    """
    if len(arr) < k:
        return False
    arrLast = arr[-1]
    for i in xrange(1,k+1):
        diff = arrLast - arr[-i]
        val = np.all(np.abs(diff) < tol)
        if not val:
            return False
    return True


# In[19]:

def dualGap(grad,r):
    """
    dualGap(1-d numpy array, 1-d numpy array) --> float
    
    Takes as input the gradient and the bloch vector in (x,y,z) form,
    returns the surrogate duality gap
    
    Arguments:
        grad : 1-d numpy array, representing gradient of function
               with entries Bloch coordinates (x,y,z)
        r   :  1-d numpy array, representing point at which gradient 
               function is computed, with entries in Bloch coordinates (x,y,z)
     Returns
         float, representing the surrogate duality gap max_{|v| <= 1}[grad(r).(r-v)]
    
    """
    val = np.dot(grad,r)
    matGrad = np.array([[grad[2], grad[0] + 1j*grad[1]],
                        [grad[0] - 1j*grad[1], -1.*grad[2]]])
    val -= np.linalg.eigvalsh(matGrad)[0]
    return val

