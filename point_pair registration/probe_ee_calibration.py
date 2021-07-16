# Perform pivot calibration for GA probing EE
# by Xiao Huang 06/29/2021

import numpy as np
import Kinematics as Yomikin
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import fileDialog
import Readers as Yomiread

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


# AOS pivoting method
# t[0:3] is the dynamic reference frame
# t[3:6] is the pivoting point in world frame
def AOS_pivot_calibration(joint, kdl):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    m, n = np.shape(joint_tem)
    A = np.zeros((3 * m, 6))
    b = np.zeros(3 * m)
    for i in range(m):
        joint_test = np.insert(joint_tem[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_test)
        t_tem = np.matmul(pos[7], [0, 0, 0, 1])
        R_tem = pos[7]
        A[3 * i, :] = np.append(R_tem[0, 0:3], -np.array([1, 0, 0]))
        A[3 * i + 1, :] = np.append(R_tem[1, 0:3], -np.array([0, 1, 0]))
        A[3 * i + 2, :] = np.append(R_tem[2, 0:3], -np.array([0, 0, 1]))
        b[3 * i] = t_tem[0]
        b[3 * i + 1] = t_tem[1]
        b[3 * i + 2] = t_tem[2]
    t = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), -b)
    return t[0:3], t[3:6]


# select a testing log file
test_log = fileDialog.select_file()
ta_joint_angles = Yomiread.readLog_tracker(test_log)

# Re-calculate tracker kdl

