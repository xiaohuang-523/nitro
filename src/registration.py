# Registration functions for 3D
# by Xiao Huang @ 07/02/2021
# Reference1 (DIPY): https://www.dipy.org/documentation/1.0.0./examples_built/affine_registration_3d/
# Reference2 (SimpleElastix): https://simpleelastix.readthedocs.io/AffineRegistration.html

import numpy as np
import Kinematics as Yomikin
import scipy
from scipy.optimize import minimize, rosen, rosen_der
from scipy import linalg

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
    #yomi_p = p[0:4]
    yomi_FK_base_matrix = Yomikin.Yomi_Base_Matrix(yomi_p)
    hxy = p[6]
    hxz = p[7]
    hyx = p[8]
    hyz = p[9]
    hzx = p[10]
    hzy = p[11]
    #hzx = 0
    #hzy = 0
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


def rigid_registration(initial_p, static_landmarks, moving_landmarks):
    p = initial_p

    p_new, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror_rigid, p,
                                                         (static_landmarks, moving_landmarks), ftol=1e-20,
                                                         full_output=True)
    print('initial p is', p)
    print('optimized p is', p_new)
    return p_new


def looperror_rigid(p, static_landmarks, moving_landmarks):
    # get the number of landmarks
    n_points = static_landmarks.shape[0]
    err = np.zeros(n_points)
    for x in range(n_points):
        # prepare for homo transformation
        dicom_tem = np.insert(static_landmarks[x,:], 3, 1.)
        stl_tem = np.insert(moving_landmarks[x,:], 3, 1.)
        eepos = dicom_tem[0:3] - np.matmul(Yomikin.Yomi_Base_Matrix(p), stl_tem)[0:3]
        #eepos = eepos[0:3]
        err[x] = np.linalg.norm(eepos)
        #print('eer is', err[x])
    print('rms is', np.sqrt(np.sum(err ** 2)/n_points))
    return err


def affine_registration_shear(initial_p, p_rigid, static_landmarks, moving_landmarks):
    p = initial_p
    p_new, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror_shear, p,
                                                         (p_rigid, static_landmarks, moving_landmarks), ftol=1e-10,
                                                         full_output=True)
    print('initial p is', p)
    print('optimized p is', p_new)
    return p_new


def looperror_shear(p, p_rigid, static_landmarks, moving_landmarks):
    # get the number of landmarks
    n_points = static_landmarks.shape[0]
    err = np.zeros(n_points)
    for x in range(n_points):
        # prepare for homo transformation
        dicom_tem = np.insert(static_landmarks[x,:], 3, 1.)
        stl_tem = np.insert(moving_landmarks[x,:], 3, 1.)
        T = np.matmul(Yomikin.Yomi_Base_Matrix((p_rigid)), get_affine_matrix_shear(p))
        eepos = dicom_tem[0:3] - np.matmul(T, stl_tem)[0:3]
        #eepos = eepos[0:3]
        err[x] = np.linalg.norm(eepos)
        #print('eer is', err[x])
    print('rms is', np.sqrt(np.sum(err ** 2)/n_points))
    return err


def get_affine_matrix_shear(p):
    #yomi_p = p[0:6]
    #yomi_FK_base_matrix = Yomikin.Yomi_Base_Matrix(yomi_p)
    hxy = p[0]
    hxz = p[1]
    hyx = p[2]
    hyz = p[3]
    hzx = p[4]
    hzy = p[5]
    shear_mtx = np.array([[1, hxy, hxz, 0],
                          [hyx, 1, hyz, 0],
                          [hzx, hzy, 1, 0],
                          [0, 0, 0, 1]])
    return shear_mtx


# point set registration
# pc1 and pc2 are list of points (could be 2d or 3d points)
# pc2 = R * pc1 + t
def point_set_registration(pc1, pc2):
    pc1_mean = np.mean(np.asarray(pc1), 0)
    pc2_mean = np.mean(np.asarray(pc2), 0)
    pc1_c = np.asarray(pc1) - pc1_mean
    pc2_c = np.asarray(pc2) - pc2_mean
    U,s,V = np.linalg.svd(np.matmul(np.transpose(pc1_c), pc2_c))
    V = np.transpose(V)
    det = np.linalg.det(np.matmul(V,U))
    R = np.matmul(V, np.matmul(np.diag([1, 1, det]), np.transpose(U)))
    t = pc2_mean - np.matmul(R, pc1_mean)
    return R, t