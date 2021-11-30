import numpy as np 
from gs_functions import *

epsilon = 1e-5

x1 = np.array([1.1, 0.9, 1])
x2 = np.array([2.2, 1.85, 1.2])
z  = np.array([0.9, 1.1, 1.05])

# get the analytic Jacobian
e, A, B = linearize_pose_pose_constraint(x1, x2, z)

print('error')
print(e)
print('A')
print(A)
print('B')
print(B)
#check the error vector
e_true = np.array([-1.06617,  -1.18076,  -0.85000]).reshape(3,1)

if(np.linalg.norm(e-e_true) > epsilon):
  print('Your error function seems to return a wrong value')
  print('Result of your function') 
  print(e)
  print('True value')
  print(e_true)
else:
  print('The computation of the error vector appears to be correct')


# compute it numerically
delta = 0.000001
scalar = 1 / (2*delta)
# test for x1
ANumeric = np.zeros([3,3])
for d in range(3):
    curX = np.copy(x1)
    curX[d] = curX[d] + delta
    err,_,__ = linearize_pose_pose_constraint(curX, x2, z)
    curX = np.copy(x1)
    curX[d] = curX[d] - delta
    errr,__,_ = linearize_pose_pose_constraint(curX, x2, z)
    err = err - errr
    ANumeric[:,d] = (err*scalar).reshape(1,3) 



diff = ANumeric - A
if np.amax(abs(diff)) > epsilon:
  print('Error in the Jacobian for x1')
  print('Your analytic Jacobian')
  print(A)
  print('Numerically computed Jacobian')
  print(ANumeric)
  print('Difference')
  print(diff)
else:
  print('Jacobian for x1 appears to be correct')

# test for x1
BNumeric = np.zeros([3,3])
for d in range(3):
  curX = np.copy(x2)
  curX[d] = curX[d] + delta
  err,_,__ = linearize_pose_pose_constraint(x1, curX, z)
  curX = np.copy(x2)
  curX[d] = curX[d] - delta
  e,__,_ = linearize_pose_pose_constraint(x1, curX, z)
  err = err - e
  BNumeric[:,d] = (err*scalar).reshape(1,3) 

diff = BNumeric - B
if np.amax(abs(diff)) > epsilon:
  print('Error in the Jacobian for x2')
  print('Your analytic Jacobian')
  print(A)
  print('Numerically computed Jacobian')
  print(ANumeric)
  print('Difference')
  print(diff)
else:
  print('Jacobian for x2 appears to be correct')
