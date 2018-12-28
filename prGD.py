
# coding: utf-8

# In[1]:

#Aim: Implement Projected gradient descent over qubit density operators using 
#     backtrack stepsize.
#Author: Vikesh Siddhu
#        Department of Physics,
#        Carnegie Mellon University,
#        Pittsburgh, PA, USA
#Date  : 16th Jan, 2018

import prGDQubitFun as prGDFun
import numpy as np
import time as time

from scipy.optimize import minimize_scalar


# In[2]:

def doPrGDVec(fun, x0, grdFun, maxIter = 15000, grdNrmMax = 1e-8, minIter = 5, 
             stepMax = 10.0, **kargs):
    """
    
    Performs projected gradient descent over qubit density matrices.
    Uses bloch sphere vector as a variable, does backtracking 
    to decide step size. 
    
    Arguments:
        fun         :   Vector argument, real valued function
        x0          :   Starting bloch vector for the function.
        grdFun      :   Gradient function, that accpets the same arguments as fun
        stepMax     :   Maximum step size for backtracking.
        maxIter     :   Maximum number of iterations for the algorithm
        grdNrmMax   :   The minimum value of the gradient norm
        minIter     :   Minimum number of iterations
        **kargs     :   Miscellaneous keyword arguments for fun and gradFun
   
   Returns:
       A solution class with attributes,
       
        listVal     :   list of function evaluations
        listGrd     :   list of gradient evaluations
        noIter      :   the number of iterations
        pts         :   list of points
        status      :   0, if converged, 1 otherwise  
    
    Version:
        16th Jan'17
    """
    
    def projStep(x, grad):
        f2Val = fun(x, **kargs)
        (t, bet) = (stepMax, .75) 
        bt  = x - t*grad
        x1  = prGDFun.PSDProjBloch(bt)
        f1  = fun(x1, **kargs)
        f2  = f2Val
        Gt  = (x - x1)/t
        f2  -= t*np.dot(grad, Gt)
        f2  += (t/2.)*np.dot(Gt, Gt)
        while (f1 > f2):
            t   = t*bet
            bt  = x - t*grad
            x1  = prGDFun.PSDProjBloch(bt)
            f1  = fun(x1, **kargs)
            f2  = f2Val
            Gt  = (x - x1)/t
            f2  -= t*np.dot(grad, Gt)
            f2  += (t/2.)*np.dot(Gt, Gt)        
        return (x1, Gt)
    
#Precompute additional variables
#List of results
    pts     = []
    grd     = []
    fVals   = []
    status  = 1
    
#Add initial results
    
    newPt = prGDFun.PSDProjBloch(x0)
    pts += [newPt]
    step = 0
        
    while (step < maxIter):
        step += 1
        fVals += [fun(newPt, **kargs)]
        gd     = grdFun(newPt, **kargs)
        (newPt, dProj)  = projStep(newPt, gd)
        grd   += [dProj]
        pts     += [newPt]

        if np.linalg.norm(grd[-1]) < grdNrmMax :
            status = 0
            break
        
        if prGDFun.arrNoChange(pts, k = 5, tol = 1e-10):
            #print 'GD stuck'
            break

#Construct solution structure
    class Struct: pass
    sol = Struct()
    sol.listVal = fVals
    sol.listGrd = grd
    sol.noIter = step
    sol.pts = pts
    sol.status = status
    return sol   

