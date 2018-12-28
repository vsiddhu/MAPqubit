#Aim: Simulate the X,Y,Z pauli measurement on a generalized amplitude and phase damped qubit 
#     and perform reconstruction on simulated data using MAP and MLE estimators.

# Details: Chooses 'nPoints' number of states uniformly on the bloch sphere 
# Pass each state, through a generalized amplitude damping channel, and phase
# damping channel simulate 'nMes' measurements in each of the X,Y,Z pauli measurements.
# From the measurement data construct MAP and MLE estimates. 
# Store various parameters.

#Author: Vikesh Siddhu,
#        Department of Physics,
#        Carnegie Mellon University,
#        Pittsburgh, PA, USA

#Date: 6th Dec'18


# In[29]:

#Get numpy library
import numpy as np
np.seterr(all='raise')
#Get library for parallel processing
from multiprocessing import Process, Queue
import multiprocessing as mp


# In[30]:

#Import code for doing projected gradient descent
from prGD import doPrGDVec as solver
#Import code that contains function and gradient
import gadPdFun as inp


# In[31]:

import copy as cp
import time as time


# In[32]:

#Import library for storing data 
from astropy.table import Table


# In[33]:

#Import functions to compute surrogate duality gap
from prGDQubitFun import dualGap


# In[34]:

#Import functions for generating measurement data
import qbitM as qbitM                                                          
import randomRho as randRho 


# In[35]:

#Function for finding a good starting point for
#projected gradient optimization 
def goodStartPt(nMes, dAt, pVal, gma, lmd):
    """
    goodStartPt(1-d numpy array, float) --> 1-d numpy array
    
    Takes as input number of pauli measurement, pauli measurement
    data and parameters for the Generalized Amplitude Damping (GAD) and Phase
    Damping (PD) channels, returns a vector which represents the 
    unconstrained minimum of the logLGadPD
    
    
    Arguments:
        nMes   :  The number of pauli measurements to be performed
        dAt    :  1-d numpy array of length 3, with entries representing
                  number of times we get a +1 outcome for a pauli measurement.
        pVal   :  GAD parameter, probability of |0> at t = infty  
        gma    :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd    :  PD parameter, decay factor 1 - e^(-t/T2)
    
    Returns:
        xStar  : 1-d numpy array of length 3, representing
                 unconstrained minimum of the logLGadPD

    """
    #Construct good starting point for optimization
    
    tz = gma*(2.*pVal-1.)
    (lm, lmZ) = (np.sqrt((1.-lmd)*(1.-gma)), 1.-gma)
    
    pInv = 1.*dAt/nMes
    r0 = 2.*pInv - np.array([0.,0.,tz]) - 1.
    
    r0 = r0/np.array([lm,lm,lmZ])
    return r0


# In[36]:

#Function for constucting the MAP estimate under generalized
#amplitude and phase damping noise
def doMAPreconstruction(nMes, dAt, pVal, gma, lmd):
    """
    doMAPreconstruction(1-d numpy array, float) --> 1-d numpy array
    
    Takes as input number of pauli measurement, pauli measurement
    data and parameters for the Generalized Amplitude Damping (GAD) and Phase
    Damping (PD) channels. Performs a MAP reconstruction.
    
    Calls:
        doPrGDVec as solver from prGD
        depolFunc as inp
    
    Arguments:
        nMes   :  The number of pauli measurements to be performed
        dAt    :  1-d numpy array of length 3, with entries representing
                  number of times we get a +1 outcome for a pauli measurement.
        pVal   :  GAD parameter, probability of |0> at t = infty  
        gma    :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd    :  PD parameter, decay factor 1 - e^(-t/T2)
    
    Returns:
        xStar  : 1-d numpy array of length 3, representing 
                 MAP reconstructed bloch vector.

        flag   : 0 of reconstruction successful, 1 otherwise
    """
    flag = 0
    x0 = goodStartPt(nMes, dAt, pVal, gma, lmd)
    sols = []
    gdps = []
    #Halting parameter
    grdNrm = 1e-7
    
    stepSize = [.001, .002, .005, .1]
    for step in stepSize:
        sol0 = solver(inp.logLGadPD, x0, grdFun = inp.gradlogLGadPD, maxIter = 500, minIter = 0, 
             grdNrmMax = grdNrm, stepMax = step, nMeasure = nMes, data = dAt, 
             p = pVal, gma = gma , lmd = lmd)
    
        gdp0 = np.linalg.norm(sol0.listGrd[-1])
    
        sols += [sol0]
        gdps += [gdp0]

        if gdp0 < grdNrm:
            return (sol0.pts[-1], flag)
 
    index = np.argmin(gdps)
    sol = sols[index]
    pGd = gdps[index]

    xStar = sol.pts[-1]
    grad = inp.gradlogLGadPD(xStar, nMeasure = nMes, data = dAt, 
             p = pVal, gma = gma , lmd = lmd)
    #If not solved to high accuracy
    
    dGap = dualGap(grad, xStar)
    if dGap > 1e-6 and pGd > 1e-6:
        flag = 1
        #Developer Use only:
        #This part of code is for diagnosing problems with the convergence
        #of projected gradient descent
        #print 'Norm of reconstructed vector = ', np.linalg.norm(xStar)
        #print 'Projected Gradient at reconstructed vector = ', np.linalg.norm(pGd)
        #print 'Gradient at reconstructed vector = ', np.linalg.norm(grad)
        #print 'Duality gap at reconstructed point', dGap
        #print 'nMes = ', nMes
        #print 'data = ', dAt
        #print 'pVal = ', pVal
        #print 'gamma = ', gma
        #print 'lmd = ', lmd
    return (xStar, flag)


# In[37]:

#Function for taking a Bloch vector and returning its noisy version
def genNoisyVec(vecActual, pVal, gma, lmd):
    """
    genNoisyVec(list, float, float, float) --> 1-d numpy array
    
    Takes as input a list of length 3 representing a Bloch vector,
    and parameters for the Generalized Amplitude Damping (GAD) and Phase
    Damping (PD) channels, returns the action of the full channel
    on the Bloch vector
    
    Arguments:
        vecActual :  Bloch vector for the quantum state.
        pVal      :  GAD parameter, probability of |0> at t = infty  
        gma       :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd       :  PD parameter, decay factor 1 - e^(-t/T2)
    
    Returns:
        1-d numpy array representing Bloch vector
    """
    
    vecNoisy = cp.deepcopy(vecActual)
    (x,y,z) = vecNoisy
    x = x*np.sqrt((1.-lmd)*(1.-gma))
    y = y*np.sqrt((1.-lmd)*(1.-gma))
    z = gma*(2.*pVal-1.) + z*(1. - gma)
    return np.array([x,y,z])


# In[38]:

#Generate a random qubit state, generate measurement data on
#its noisy version, do MAP and MLE.
def stateReconst(nMes, pVal, gma, lmd):
    """
    stateReconst(int, float, float,
    float) -->  (1-d numpy array, 1-d numpy array, 1-d numpy array,
                 1-d numpy array, 1-d numpy array)
    
    Takes as input the number of Pauli Measurements and the amplitude damping parameter.
    Generated a uniformly random quantum state, returns a MAP and MLE reconstructed 
    quantum state and the simulated result for the pauli X,Y and Z measurements.
    
    Calls:
        numpy     as np
        qbitM     as qbitM      
        randRho   as randRho

    Arguments:
        nMes      :  The number of pauli measurements to be performed
        pVal      :  GAD parameter, probability of |0> at t = infty  
        gma       :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd       :  PD parameter, decay factor 1 - e^(-t/T2)
                  
    Returns:
        vecActual :  Bloch vector for the quantum state.
        rMAP       : Bloch vector for the MAP reconstructed state.
        rMLE       : Bloch vector for the MLE reconstructed state.
        pauliData  : A 1-D numpy array of length 3, representing
                     number of +1 counts for (X,Y,Z) pauli measurements.  
    
    """
    #Generate random vector on Bloch Ball
    vecActual = randRho.randomUnitBallPoint()
    
    #Generate noisy vector under amplitude damping noise
    vecNoisy = genNoisyVec(vecActual, pVal, gma, lmd) 
    
    #Data accumulated for measurements
    dAt = qbitM.pauliBasisMeasure(vecNoisy, nMes)
    
    #Do MAP reconstruction
    (rMAP, flagMAP) = doMAPreconstruction(nMes, dAt, pVal, gma, lmd)
    #Developer use only
    #if flagMAP:
    #    print "MAP failed at vecActual = ", vecActual
    #    print "\n"

    #Do MLE reconstruction
    (rMLE, flagMLE) = doMAPreconstruction(nMes, dAt, 0.,0.,0.)
    #Developer use only
    #if flagMLE:
    #    print "MLE failed at vecActual = ", vecActual
    #    print "\n"

    return (vecActual, rMAP, rMLE, dAt)


# In[39]:

def performExp(nMin, nMax, nTotal, pVal, gma, lmd, nPoints, fileName):
    """
    performExp(int, int, int, float, float, float, int, str) --> None
    
    Takes as input the Generalized Amplitude Damping (GAD) parameters, Phase
    Damping (PD) parameter, and the number of points and a filename. 
        
    Calls:
        randomRho as randRho
        
    Arguments:
        nMin     :  Minumum number of measurements
        nMax     :  Maximum number of measurements
        nTotal   :  Total different measurement points
        pVal     :  GAD parameter, probability of |0> at t = infty  
        gma      :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd      :  PD parameter, decay factor 1 - e^(-t/T2)
        nPoints  :  Number of points
        fileName :  Name of file to store the output


    For nTotal different measurements uniformly spaced between [nMin, nMax],
    performs reconstruction for 'nPoints' uniformaly random 
    points. For each point do MAP and MLE reconstruction
    and creates two .hdf5 files using the given 'fileName' as prefix
    and 'Res' and 'Prm' as suffix. In 'Res' suffix store an
    astropy data-table with entries
        
        dNo   : Serial number of the state.  
        r     : Bloch vector for the random state.
        MAP   : Bloch vector for the MAP reconstructed state.
        MLE   : Bloch vector for the MLE reconstructed state.
        data  : A 1-D numpy array of length 3, representing number 
                of +1 counts for (X,Y,Z) pauli measurements.  
        noMes : Number of measurements for the data.
    

    and in 'Prm' suffix store an astropy data-table with entries
    
        pVal     :  GAD parameter, probability of |0> at t = infty  
        gma      :  GAD parameter, decay factor 1 - e^(-t/T1)
        lmd      :  PD parameter, decay factor 1 - e^(-t/T2)
    """

    dNo = []
    r = []
    MAP = []
    MLE = []
    data = []
    noMes = []
    
    
    for nMes in np.linspace(nMin,nMax,nTotal):
        nMes = int(nMes)
        nProc = 4
        pool = mp.Pool(processes=nProc)
        
        prmVal = (nMes, pVal, gma, lmd)
        res = [pool.apply_async(stateReconst, args = prmVal) for j in xrange(nPoints)]
        
        pool.close()
        pool.join()
        
        res = [p.get() for p in res]
        
        for i in xrange(nPoints):
            dNo += [i]
            r += [res[i][0]]
            MAP += [res[i][1]]
            MLE += [res[i][2]]
            data += [res[i][3]]
            noMes += [nMes]
    
    #Save the data
    
    tab1 = [dNo, r, MAP, MLE, data, noMes]
    table1 = Table(tab1, names = ['Dno', 'r', 'MAP', 'MLE', 'data', 'noMes'])
    
    tab2 = [[pVal], [gma], [lmd]]
    table2 = Table(tab2, names = ['pVal', 'gma', 'lmd'])
    
    name = fileName + 'Res' + '.hdf5'
    table1.write(name, path ='/data')
    
    name = fileName + 'Prm' + '.hdf5'
    table2.write(name, path ='/data')


# In[40]:

def runSimulation(nMin, nMax, nTotal, nPoints, pVal, k, T1, T2):
    """
    runSimulation(int, int, int, int, float, int, float, float) --> None
    
    For nTotal different measurements uniformly spaced between [nMin, nMax],
    for nPoints number of qubit states, passing through a generalized amplitude
    damping channel with parameters pVal and T1 and phase damping channel
    with parameter T2, for a time interval k*T2 prior to measurement do MAP
    and MLE reconstruction using 'performExp' function.
    
    The 'performExp' function and stores result in 
    a hdf5 file with name 'res' + 'k'. See 'performExp' docstring
    for details
    
    Arguments:
        nMin     :  Minumum number of measurements
        nMax     :  Maximum number of measurements
        nTotal   :  Total different measurement points
        nPoints  :  Number of points
        pVal     :  GAD parameter, probability of |0> at t = infty  
        k        :  Factor of T2 Time for which channels act
        T1       :  GAD parameter, Spin-Lattice relaxation
        T2       :  PD parameter, Spin-Spin relaxation
        
    Returns:
        None
    """
    
    filename = 'res' + str(k)
    #print filename
    t = T2*k
    gma = 1. - np.exp(-t/T1)
    lmd = 1. - np.exp(-t/T2)
    performExp(nMin, nMax, nTotal, pVal, gma, lmd, nPoints, filename)



pVal    = 0.5
T1      = 5.
T2      = T1/10.
nPoints = 25000
nMin    = 10
nMax    = 2000
nTotal  = 20

tSt = time.time()

kVals = [0.25,.5,.75,1.0,1.5,2.0,2.5]
for k in kVals:
    runSimulation(nMin, nMax, nTotal, nPoints, pVal, k, T1, T2)



# In[13]:

tEn = time.time()
print 'total time = ', tEn - tSt

