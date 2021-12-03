import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import inv
from scipy.sparse.linalg import lsqr

def t2v(A):
    """
    Homogeneous transformation to vector
    A = H = [R d
             0 1]
             
    Rotation matrix:
    R = [+cos(theta), -sin(theta)
         +sin(theta), +cos(theta)]
         
    translation vector:
    d = [x y]'
    """
    v = np.zeros((3,1), dtype = np.float64)
    
    v[0] = A[0,2] # x
    v[1] = A[1,2] # y
    v[2] = np.arctan2(A[1,0], A[0,0]) # theta
    
    return v
    
def v2t(v):
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
    x = v[0]
    y = v[1]
    theta = v[2]
    
    A = np.array([[+np.cos(theta), -np.sin(theta), x],
                  [+np.sin(theta), +np.cos(theta), y],
                  [             0,              0, 1]], dtype=np.float64)
    
    return A

def solve(H, b, sparse_solve):
    """
    Solve sparse linear system H * dX = -b
    """            
    if sparse_solve:
        # # Transformation to sparse matrix form
        H_sparse = csr_matrix(H) 
        # Solve sparse system
        dX = spsolve(H_sparse, b)

    else:    
        # Solve linear system
        dX = np.linalg.solve(H, b)
        
    # Keep first node fixed    
    # dX[:3] = [0, 0, 0] 
    
    # Check NAN
    dX[np.isnan(dX)] = 0
    
    return dX

def linearize_pose_pose_constraint(x_i, x_j, z_ij):
    zt_ij = v2t(z_ij.reshape(3,1)) 
    xt_i  = v2t(x_i.reshape(3,1))
    xt_j  = v2t(x_j.reshape(3,1))

    R_i  = xt_i[:2,:2]
    R_ij = zt_ij[:2,:2] 

    ct_i = np.cos(x_i[2])
    st_i = np.sin(x_i[2])
    # Derivative of R_i with respect to theta_i
    dR_i = np.array([[-st_i, ct_i],
                     [-ct_i, -st_i]])
    


    # e_ij = t2v(np.dot(np.dot(inv(zt_ij), inv(xt_i)), xt_j))
    e_ij = t2v(np.dot(inv(zt_ij), np.dot(inv(xt_i), xt_j)))
    
    A_ij_11_block = np.dot(-R_ij.T, R_i.T)

    deltaX = (x_j - x_i)[:2].reshape(2,1)  #(x_j - x_i)
    
    A_ij_12_block = np.dot(R_ij.T, np.dot( dR_i, deltaX ))
    A_ij_21_22_block = np.array([0, 0, -1])
    
    A_ij = np.vstack((np.hstack((A_ij_11_block, A_ij_12_block)),A_ij_21_22_block))


    B_ij_11_block = np.dot(R_ij.T, R_i.T)
    B_ij_12_block = np.zeros((2,1), dtype = np.float64)
    B_ij_21_22_block = np.array([0, 0, 1])
    
    B_ij = np.vstack((np.hstack((B_ij_11_block, B_ij_12_block)),B_ij_21_22_block))
        
    return e_ij, A_ij, B_ij

def linearize_pose_landmark_constraint(x_i, x_j, z_ij):
    xt_i = v2t(x_i)
    xt_j = x_j.reshape(2,1)
    R_i = xt_i[:2,:2]
    t_i =xt_i[:2,2].reshape(2,1)
    t_m = xt_j.reshape(2,1)
    
    # l = np.hstack([x_j, 1])
    # z = np.hstack([z_ij,1])

    t_im = np.array([z_ij]).reshape(2,1)
    #error function
    e_ij =(np.dot(R_i.T,(t_m - t_i)) - t_im)
    # Derivative of R_i with respect to theta_i
    dR_i = np.array([[-np.sin(x_i[2]), np.cos(x_i[2])],
                    [-np.cos(x_i[2]), -np.sin(x_i[2])]])

    A_ij_block_11 = -R_i.T
    A_ij_block_12 = np.dot(dR_i,(t_m - t_i)) 

    A_ij = np.hstack([A_ij_block_11, A_ij_block_12])  # 2x3 matrix
    B_ij = R_i.T

    return e_ij, A_ij, B_ij