import numpy as np
from gs_classes import *

def vec2tran(pose):
    """
    Vector to Homogeneous transformation
    A = H = [R d
             0 1]
             
    Rotation matrix:
    R = [+cos(theta), -sin(theta)
         +sin(theta), +cos(theta)]
         
    translation vector:
    d = [x y]'
    """
    x = pose[0]
    y = pose[1]
    theta = pose[2]
    
    A = np.array([[+np.cos(theta), -np.sin(theta), x],
                  [+np.sin(theta), +np.cos(theta), y],
                  [             0,              0, 1]])
    
    return A


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """"
    Calculate covariance ellipse , angle in degrees
    """
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    return width, height, theta


def dataAssociationPoints(list, measurement, pose, cov=None, nstd=2):
    """
    Returns index of matching/already seen point else it returns -1
    """
    #calculate measurement within world space
    pose = [pose.x, pose.y, pose.theta]
    T_ = vec2tran(pose) 
    measurement = np.array([measurement[0], measurement[1], 1])
    m_ = np.dot(T_, measurement)

    #calculate ellipse parameters from covariance
    if cov is None:
        cov = np.eye(2)*0.3  #Arbetrary covariance cicle of .1

    width, height, theta = cov_ellipse(cov,nstd)
    theta = -np.radians(theta)

    #check if measurement lies within the covariance 
    index = -1
    for i in range(len(list)):
        point = list[i]
        x_ = m_[0] - point.x
        y_ = m_[1] - point.y
        a = width /2
        b = height /2
        val = (((x_*np.cos(theta) - y_*np.sin(theta))**2) / a**2) + (((x_*np.sin(theta) + y_*np.cos(theta))**2) / b**2) 
        if val <=1:
            index = i 
            break
    
    return index , m_
