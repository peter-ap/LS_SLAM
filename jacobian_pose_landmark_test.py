from typing_extensions import Annotated
import numpy as np
from gs_functions import *

epsilon = 1e-5

x1 = np.array([1.1, 0.9, 1])
x2 = np.array([2.2, 1.9])
z  = np.array([1.3, -0.4])

e, A_ij, B_ij = linearize_pose_landmark_constraint(x1, x2, z)

#check the error vector 
e_true = np.array([0.135804, 0.014684]).reshape(2,1)

if(np.linalg.norm(e-e_true) > epsilon):
    print("your error function seems to return a wrong value")
    print("Result of your function= ", e)
    print("true value is = ", e_true)
else:
    print("computation of the error vector appears to be correct")


#compute it numarically 
delta = 1e-6

scalar = 1 / (2*delta)
# test for x1
ANumeric = np.zeros([2,3])
for d in range(3):
  curX = np.copy(x1)
  curX[d] = curX[d] + delta
  err, ww, zz = linearize_pose_landmark_constraint(curX, x2, z)
  curX = np.copy(x1)
  curX[d] = curX[d] - delta
  e, ww, zz = linearize_pose_landmark_constraint(curX, x2, z)
  err = err - e
  ANumeric[:, d] = (err * scalar).reshape(1,2)



diff = ANumeric - A_ij
if np.amax(abs(diff)) > epsilon:
    print('Error in the Jacobian for x1')
    print("Your analytic Jacobian ")
    print(A_ij)
    print('numerically computed Jacobian')
    print(ANumeric)
else:
    print('Jacobian for x1 appears to be correct')


