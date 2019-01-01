import numpy as np


# In[5]:

def pauliBasisMeasure(vec, trial):
    """
    pauliBasisMeasure(1-d numpy array, int) --> 1-d numpy array
    
    Takes a Bloch vector and number of trials for each Pauli measurement.
    Returns a string with number of +1 counts for (X,Y,Z) measurements
    
    Arguments:
        vec   :  Bloch Vector for the density operator
        trial :  Number of trials to be performed for each measurement 
    
    Returns
        (valX, valY, valZ) = The number of +1 counts for each X,Y,Z measurement
    """
    
    n = trial 
    p = (1. + vec)/2.
    np.random.seed()
    return np.random.binomial(n, p)

