import numpy as np
import copy as cp


def logLGadPD(vec, nMeasure, data, p, gma, lmd):
    """
    logLGadPD(1-d numpy array, int, 1-d numpy array, 
    float, float, float) --> float
    
    Takes as input a bloch vector, the number of pauli measurements, number
    of times each (x,y,z) measurement came +1 and parameters for the 
    Generalized Amplitude Damping (GAD) and Phase Damping (PD) channels,
    returns the log likelohood function.
    
    Arguments:
        vec      : Bloch vector for density operator, 1-d numpy array
                   of length 3.
        nMeasure : Number of measurements
        data     : 1-d numpy array of length 3, with entries representing
                   number of times we get a +1 outcome for a pauli measurement.
        p        :  GAD parameter, probability of |0> at t = infty  
        gma      :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd      :  PD parameter, decay factor 1 - e^(-t/T2)
    
    Returns:
        -ve - log-likelihood function
    """
    
    #GAD-PD shrinking
    vecNoisy = cp.deepcopy(vec)
    (x,y,z) = vecNoisy
    x = x*np.sqrt((1.-lmd)*(1.-gma))
    y = y*np.sqrt((1.-lmd)*(1.-gma))
    z = gma*(2.*p-1.) + z*(1. - gma)
    rVec = np.array([x,y,z])
    
    pVec = (rVec + 1.0)/2.
    
    #Avoiding the 0. and 1. 
    pVec[pVec == 0.] = 1e-14
    pVec[pVec == 1.] = 1.- 1e-14

    val = data*np.log(pVec) + (nMeasure - data)*np.log(1.-pVec)
    return -np.sum(val)


def gradlogLGadPD(vec, nMeasure, data, p, gma, lmd):
    """
    gradlogLGadPD(1-d numpy array, int, 1-d numpy array, float) --> 1-d numpy array
    
    Takes as input a bloch vector, the number of pauli measurements, number
    of times each(x,y,z) measurement came +1 and parameters for the 
    Generalized Amplitude Damping (GAD) and Phase Damping (PD) channels,
    returns the gradient of log likelohood function.
    
    Arguments:
        vec      : Bloch vector for density operator, 1-d numpy array
                   of length 3.
        nMeasure : Number of measurements
        data     : 1-d numpy array of length 3, with entries representing
                   number of times we get a +1 outcome for a pauli measurement.
        p        :  GAD parameter, probability of |0> at t = infty  
        gma      :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd      :  PD parameter, decay factor 1 - e^(-t/T2)
    
    Returns:
        Gradient of the -ve - log-likelihood function w.r.t bloch
        vector.
    """
    #GAD-PD shrinking
    vecNoisy = cp.deepcopy(vec)
    (x,y,z) = vecNoisy
    x = x*np.sqrt((1.-lmd)*(1.-gma))
    y = y*np.sqrt((1.-lmd)*(1.-gma))
    z = gma*(2.*p-1.) + z*(1. - gma)
    rVec = np.array([x,y,z])
    
    pVecPlus = (rVec + 1.0)/2.

    #Avoiding the 0. and 1. 
    pVecPlus[pVecPlus == 0.] = 1e-14
    pVecPlus[pVecPlus == 1.] = 1.- 1e-14
    
    pVecMins = 1. - pVecPlus
    
    nVecPlus = data
    nVecMins = nMeasure - data
    
    val = (nVecPlus/pVecPlus) - (nVecMins/pVecMins) 
    val[0] = val[0]*np.sqrt((1.-lmd)*(1.-gma))/2.
    val[1] = val[1]*np.sqrt((1.-lmd)*(1.-gma))/2.
    val[2] = val[2]*(1. - gma)/2.
    return -val

###############################################################################
#Developer Use only:
###############################################################################

## In[20]:
#
#import qbitM as qbitM
#import randomRho as randRho
#from scipy.optimize import check_grad
#
#
## In[21]:
#
#for i in xrange(20):
#    nMeasure = 10
#    vecActual = randRho.randomUnitBallPoint()
#    vecActual = np.array(vecActual)
#    data = qbitM.pauliBasisMeasure(vecActual, nMeasure)
#    p = .13
#    gma = .1
#    lmd = .21
#    print check_grad(logLGadPD, gradlogLGadPD, vecActual, nMeasure, data, p, gma, lmd)

###############################################################################

