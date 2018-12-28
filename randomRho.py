
# coding: utf-8

# In[50]:

import numpy as np


# In[51]:

def randomSphereUniformRadius():
    """
    randomSphereUniformRadius() --> 1-d list
    
    Takes no input and returns a uniformly random
    point in shell with radius chosen uniformly between
    [0,1].
    
    Arguments:
        None
    
    Returns:
        Cartesian coordinated of a point chosen
        uniformly at random in a 3D Ball
    
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    r = np.random.uniform(0,1.) 
    
    theta = np.arccos( costheta )
    
    x = r*np.sin( theta) * np.cos( phi )
    y = r*np.sin( theta) * np.sin( phi )
    z = r*np.cos( theta )
    return (x,y,z)

def randomUnitBallPoint():
    """
    randomUnitBallPoint() --> 1-d list
    
    Takes no input and returns a uniformaly random
    point in a 3D ball of unit radius.
    
    Arguments:
        None
    
    Returns:
        Cartesian coordinated of a point chosen
        uniformly at random in a 3D Ball
    
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    l = np.random.uniform(0,1./3.) 
    r = (3*l)**(1./3) 
    
    theta = np.arccos( costheta )
    
    x = r*np.sin( theta) * np.cos( phi )
    y = r*np.sin( theta) * np.sin( phi )
    z = r*np.cos( theta )
    return (x,y,z)


# In[52]:

def randomSpherePoint(r):
    """
    randomSpherePoint(float) --> 1-d list
    
    Takes as input a radius returns a uniformaly random
    point on the shell of that radius.
    
    Arguments:
        r : Float, represeting radius
    
    Returns:
        Cartesian coordinated of a point chosen
        uniformly at random on shell or radius r
    
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    
    theta = np.arccos( costheta )
    
    x = r*np.sin( theta) * np.cos( phi )
    y = r*np.sin( theta) * np.sin( phi )
    z = r*np.cos( theta )
    return (x,y,z)

