# Affine registration for 3D
# by Xiao Huang 03/15/2021
# Reference1 (DIPY): https://www.dipy.org/documentation/1.0.0./examples_built/affine_registration_3d/
# Reference2 (SimpleElastix): https://simpleelastix.readthedocs.io/AffineRegistration.html

import numpy as np
import Kinematics as Yomikin
import scipy
from scipy.optimize import minimize, rosen, rosen_der

# Prepare affine matrix
# Affine transformation includes: translation, rotation, scale and shear.
# For the purpose of IOS and DICOM matching, only translation, rotation and shear are considered.
# The order is as follows: rotation, translation, shear
# The parameters are in the order of Tx, Ty, Tz, Rz, Ry, Rx, hxy, hxz, hyx, hyz, hzx, hzy.
# So that the Yomi FK matrix can be used.
# Input:
#       p: 1-d array with [tx, ty, tz, rz, ry, rx, hxy, hxz, hyx, hyz, hzx, hzy]
def get_affine_matrix(p):
    yomi_p = p[0:6]
    yomi_FK_base_matrix = Yomikin.Yomi_Base_Matrix(yomi_p)
    hxy = p[6]
    hxz = p[7]
    hyx = p[8]
    hyz = p[9]
    hzx = p[10]
    hzy = p[11]
    shear_mtx = np.array([[1, hxy, hxz, 0],
                          [hyx, 1, hyz, 0],
                          [hzx, hzy, 1, 0],
                          [0, 0, 0, 1]])
    T = np.matmul(yomi_FK_base_matrix, shear_mtx)
    return T


def looperror(p, static_landmarks, moving_landmarks):
    # get the number of landmarks
    n_points = static_landmarks.shape[0]
    err = np.zeros(n_points)
    for x in range(n_points):
        # prepare for homo transformation
        dicom_tem = np.insert(static_landmarks[x,:], 3, 1.)
        stl_tem = np.insert(moving_landmarks[x,:], 3, 1.)
        eepos = dicom_tem[0:3] - np.matmul(get_affine_matrix(p), stl_tem)[0:3]
        #eepos = eepos[0:3]
        err[x] = np.linalg.norm(eepos)
        #print('eer is', err[x])
    print('rms is', np.sqrt(np.sum(err ** 2)/n_points))
    return err

def affine_registration(initial_p, static_landmarks, moving_landmarks):
    p = initial_p
    p_new, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror, p,
                                                         (static_landmarks, moving_landmarks), ftol=1e-10,
                                                         full_output=True)
    print('initial p is', p)
    print('optimized p is', p_new)
    return p_new



