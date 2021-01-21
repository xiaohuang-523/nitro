import Readers as Yomiread
import Kinematics as Yomikin
import numpy as np
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import math

# Delete after verification
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import Data
import datetime

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# Define functions for PT calibrations
def unpack(kp_2b_update, p, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_update)
    phat = np.copy(p)
    kp_fit = kp_update_parameters[:]
    xx = np.nonzero(kp_fit)
    kp[xx] = phat
    k = kp[:]
    return k


def pack(kp_2b_packed, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_packed)
    kp_p = np.copy(kp_update_parameters)
    p_tem = kp[np.nonzero(kp_p)]
    p = np.array(p_tem)
    return p


def fitSphere(xx, yy, zz):
    # Exact solution for sphere fitting
    # For more details, check the literature
    # "Fast Geometric Fit Algorithm for Sphere Using Exact Solution" by Sumith YD, 2015
    x = np.copy(xx)
    y = np.copy(yy)
    z = np.copy(zz)

    N = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sz = np.sum(z)
    Sxx = np.sum(x * x)
    Syy = np.sum(y * y)
    Szz = np.sum(z * z)
    Sxy = np.sum(x * y)
    Syz = np.sum(y * z)
    Sxz = np.sum(x * z)

    Sxxx = np.sum(x * x * x)
    Syyy = np.sum(y * y * y)
    Szzz = np.sum(z * z * z)
    Sxyy = np.sum(x * y * y)
    Sxzz = np.sum(x * z * z)
    Sxxy = np.sum(x * x * y)
    Sxxz = np.sum(x * x * z)
    Syyz = np.sum(y * y * z)
    Syzz = np.sum(y * z * z)

    A1 = Sxx + Syy + Szz
    a = 2 * Sx * Sx - 2 * N * Sxx
    b = 2 * Sx * Sy - 2 * N * Sxy
    c = 2 * Sx * Sz - 2 * N * Sxz
    d = -N * (Sxxx + Sxyy + Sxzz) + A1 * Sx
    e = 2 * Sx * Sy - 2 * N * Sxy
    f = 2 * Sy * Sy - 2 * N * Syy
    g = 2 * Sy * Sz - 2 * N * Syz
    h = -N * (Sxxy + Syyy + Syzz) + A1 * Sy

    j = 2 * Sx * Sz - 2 * N * Sxz
    k = 2 * Sy * Sz - 2 * N * Syz
    l = 2 * Sz * Sz - 2 * N * Szz
    m = -N * (Sxxz + Syyz + Szzz) + A1 * Sz

    delta = a * (f * l - g * k) - e * (b * l - c * k) + j * (b * g - c * f)
    xc = (d * (f * l - g * k) - h * (b * l - c * k) + m * (b * g - c * f)) / delta
    yc = (a * (h * l - m * g) - e * (d * l - m * c) + j * (d * g - h * c)) / delta
    zc = (a * (f * m - h * k) - e * (b * m - d * k) + j * (b * h - d * f)) / delta
    R = np.sqrt(xc ** 2 + yc ** 2 + zc ** 2 + (A1 - 2 * (xc * Sx + yc * Sy + zc * Sz)) / N)
    return xc, yc, zc, R


# Estimate ball locations for general bar method
def estBall(joint):
    joint_tem = np.copy(joint)
    m, n = np.shape(joint_tem)
    EE_p_est = []
    for k in range(m):
        EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(kp_guess_common, joint_tem[k, :])
        # EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(tracker_kdl, joint_tem[k, :])
        EE_p_est_tem = np.matmul(EE_T_est_tem[7], [0, 0, 0, 1])
        # EE_p_est_tem = np.matmul(EE_T_est_tem[6], [0, 0, 0, 1])
        EE_p_est.append(EE_p_est_tem[0:3])
    EE_p_est = np.asarray(EE_p_est)
    ball = fitSphere(EE_p_est[:, 0], EE_p_est[:, 1], EE_p_est[:, 2])[0:3]
    return np.asarray(ball)


def solve_commonbase(hole1, hole2, hole3):
    s1 = np.asarray(hole1)
    s2 = np.asarray(hole2)
    s3 = np.asarray(hole3)
    y_axis = (s3 - s2) / np.linalg.norm(s3 - s2)
    z_axis = -np.cross(y_axis, s1 - s3) / np.linalg.norm(np.cross(y_axis, s1 - s3))
    x_axis = np.cross(y_axis, z_axis)
    rotM = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                     [x_axis[1], y_axis[1], z_axis[1]],
                     [x_axis[2], y_axis[2], z_axis[2]]])
    t = s1
    # Solve the inverse of Rot and t.
    rotMi = np.transpose(rotM)
    ti = -t
    return rotMi, ti

def AOS_pivot_calibration(joint, kdl):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    m, n = np.shape(joint_tem)
    A = np.zeros((3 * m, 6))
    b = np.zeros(3 * m)
    err = np.zeros((3 * m, 3))
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


def looperror(cal, jointangles, bb, ball):
    kp_tem = np.copy(kp_guess_common)
    cal_tem = np.copy(cal)
    kp_fit_tem = np.copy(kp_fit_common)
    joint_tem = np.copy(jointangles)

    cal_tem = unpack(kp_tem, cal_tem, kp_fit_tem)
    m, n = np.shape(jointangles)
    err = np.zeros(shape=(m,))
    for x in range(m):
        joint_tem_f = np.insert(joint_tem[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(cal_tem, joint_tem_f)
        # If the common reference is removed, use '6'
        # If the common reference is kept, use '7'
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball) - bb[x]

    if DEBUG_FLAG == 1:
        print('rms is', np.sqrt(np.sum(err ** 2, axis=0))/m)
    return err


def opt_bb(joint, bb, ball):
    joint_tem = np.copy(joint)
    bb_tem = np.copy(bb)
    kp_guess_common_tem = np.copy(kp_guess_common)
    kp_fit_common_tem = np.copy(kp_fit_common)
    opt = pack(kp_guess_common_tem, kp_fit_common_tem)
    ball_tem = np.copy(ball)

    if DEBUG_FLAG == 1:
        print('bb_tem before is', bb_tem)
        print('joint_tem before is', joint_tem)
        print('ball_tem before is', ball_tem)

    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror, opt,
                                                         (joint_tem, bb_tem, ball_tem), ftol=1e-20,
                                                         full_output=True)
    tem_p = p[:]
    kdl = tem_p[:]
    if DEBUG_FLAG == 1:
        print('bb_tem after is', bb_tem)
        print('joint_tem after is', joint_tem)
        print('ball_tem after is', ball_tem)
        print('integer flag is', ier)
        print('message is', mesg)
        jac = infodict["fjac"]
        jacobian_condition_number = np.linalg.cond(jac)
        print('condition number is', jacobian_condition_number)
        print('kdl is', kdl)
    ball_est = ball
    kdl_unpack = unpack(kp_guess_common_tem, kdl, kp_fit_common_tem)
    if DEBUG_FLAG == 1:
        print('kdl_unpack', kdl_unpack)
    Vbe_tem, V1e_tem, V2e_tem, count_tem = eeVector(kdl_unpack, joint_tem)
    rms = np.sqrt(np.sum((Vbe_tem - ball_tem)**2)/np.shape(joint_tem))
    print('ball bar calibration rms is', rms)
    return kdl_unpack

def opt_full(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3, joint_s1, joint_s2, joint_s3):
    print('Optimizing with single ball')
    bb1 = np.tile(bb_1, [joint_bb1.shape[0], 1])
    bb2 = np.tile(bb_2, [joint_bb2.shape[0], 1])
    bb3 = np.tile(bb_3, [joint_bb3.shape[0], 1])

    j = np.vstack((joint_bb1, joint_bb2, joint_bb3))
    bb = np.vstack((bb1, bb2, bb3))
    ball = (estBall(joint_bb1) + estBall(joint_bb2) + estBall(joint_bb3))/3

    # Solve bb calibration
    time1 = np.datetime64('now')
    kdl_tem = opt_bb(j, bb, ball)
    kdl_bb_without_outlier_rejection = np.copy(kdl_tem)
    time2 = np.datetime64('now')
    if DEBUG_FLAG == 1:
        print('calculation time is', time2-time1)
    # Remove outliers in bb calibration
    err_data, count_data, out_indx, rms_data = find_outlier_bb(j, bb, ball, kdl_tem)
    j_2 = remove_outlier(j, out_indx)
    bb_2 = remove_outlier(bb, out_indx)
    if DEBUG_FLAG == 1:
        print('Checking ball bar calibration outliers')
        print(np.str(np.shape(j)) + ' total points were collected in ball bar calibration')
        print('Number of outliers in ball bar calibration is ' + np.str(j.shape[0] - j_2.shape[0]))
        print('Percentage of outliers is : % 2.1f ' % ((j.shape[0] - j_2.shape[0]) / j.shape[0] * 100))

    # Solve bb calibration after outlier rejection
    if REMOVE_OUTLIER_BB == 1:
        kdl_tem = opt_bb(j_2, bb_2, ball)
        kdl_bb_after_outlier_rejection = np.copy(kdl_tem)
        if DEBUG_FLAG == 1:
            print('Ball bar calibration before outlier rejection is ', kdl_bb_without_outlier_rejection)
            print('Ball bar calibration after outlier rejection is ', kdl_bb_after_outlier_rejection)
            print('Difference of ball bar calibration KDL due to outlier rejection is ',
                  kdl_bb_without_outlier_rejection - kdl_bb_after_outlier_rejection)

    # solve pivot calibrations
    s1t, s1b = AOS_pivot_calibration(joint_s1, kdl_tem)
    s2t, s2b = AOS_pivot_calibration(joint_s2, kdl_tem)
    s3t, s3b = AOS_pivot_calibration(joint_s3, kdl_tem)

    # remove outliers in pivoting
    if REMOVE_OUTLIER_PV == 1:
        s1t_2, s1b_2 = remove_outlier_pivot(joint_s1, kdl_tem, s1t, s1b)
        s2t_2, s2b_2 = remove_outlier_pivot(joint_s2, kdl_tem, s2t, s2b)
        s3t_2, s3b_2 = remove_outlier_pivot(joint_s3, kdl_tem, s3t, s3b)
        del s1t, s1b, s2t, s2b, s3t, s3b
        s1t = s1t_2
        s1b = s1b_2
        s2t = s2t_2
        s2b = s2b_2
        s3t = s3t_2
        s3b = s3b_2

    # solve common base transformation
    rot, t = solve_commonbase(s1b, s2b, s3b)
    # Estimate pivot calibration std
    rtip = np.vstack((np.vstack((s1t, s2t)), s3t))

    if DEBUG_FLAG == 1:
        std = np.std(rtip, axis=0)
        mean = np.mean(rtip, axis=0)
        print('Rtip std and mean are', std, mean)

    kdl_final = convert_Yomi_convention(kdl_tem,rot,t)

    filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename1 = 'C:\\Calibration\\tracker_kdl\\tracker_kdl_' + filename1 + '.txt'
    np.savetxt(filename1, kdl_final, newline=",\n")
    return kdl_final


def find_outlier_bb(joint, bb, ball, kdl):
    joint_tem = np.copy(joint)
    bb_tem = np.copy(bb)
    ball_tem = np.copy(ball)
    kdl_tem = np.copy(kdl)
    m, n = np.shape(joint_tem)
    err = np.zeros(shape=(m,))
    count = np.zeros(shape=(m,))
    out_indx = np.array([])
    for x in range(m):
        joint_tem_f = np.insert(joint_tem[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_tem_f)
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball_tem) - bb_tem[x]
        count[x] = x+1
        if err[x,] > 0.0001:
            out_indx = np.append(out_indx, np.array([x]))
    eer_rms = np.var(err)
    return err, count, out_indx, eer_rms


def find_outlier_pivot(joint, kdl, rtip, divot):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    rtip_tem = np.copy(rtip)
    divot_tem = np.copy(divot)
    rtip_homo = np.insert(rtip_tem,3,1)
    m, n = np.shape(joint_tem)
    err = np.zeros(shape=(m,3))
    count = np.zeros(shape=(m,))
    out_indx = np.array([])
    for x in range(m):
        joint_tem_f = np.insert(joint_tem[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_tem_f)
        eepos = np.matmul(eepos[7], rtip_homo)
        err[x,] = eepos[0:3]
        count[x] = x+1
    for x in range(m):
        if np.linalg.norm(err[x,] - divot_tem) > 0.0001:
            out_indx = np.append(out_indx, np.array([x]))
    eer_rms = np.var(err)
    return err, count, out_indx, eer_rms


def remove_outlier(array, out_indx):
    array_tem = np.copy(array)
    indx_tem = np.copy(out_indx)
    for index in sorted(indx_tem, reverse=True):
        array_tem = np.delete(array_tem, int(index), axis=0)
    return array_tem


def remove_outlier_pivot(joint, kdl, hole, divot):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    hole_tem = np.copy(hole)
    divot_tem = np.copy(divot)
    err_data, count_data, out_indx, rms_data = find_outlier_pivot(joint_tem, kdl_tem, hole_tem, divot_tem)
    joint_s1_no = remove_outlier(joint_tem, out_indx)
    s1t_no, s1b_no = AOS_pivot_calibration(joint_s1_no, kdl_tem)
    err_data2, count_data2, out_indx2, rms_data2 = find_outlier_pivot(joint_s1_no, kdl_tem, s1t_no, s1b_no)
    joint_s2_no = remove_outlier(joint_s1_no, out_indx2)
    s1t_no2, s1b_no2 = AOS_pivot_calibration(joint_s2_no, kdl_tem)
    err_data3, count_data3, out_indx3, rms_data3 = find_outlier_pivot(joint_s2_no, kdl_tem, s1t_no2, s1b_no2)
    joint_s3_no = remove_outlier(joint_s2_no,out_indx3)
    s1t_no3, s1b_no3 = AOS_pivot_calibration(joint_s3_no, kdl_tem)
    return s1t_no3, s1b_no3


def convert_homo_transform(rot,t):
    rot_tem = np.copy(rot)
    t_tem = np.copy(t)
    t1 = np.array([[0.],[0.],[0.]])
    for i in range(3):
        t1[i,0] = np.matmul(rot_tem, t_tem)[i]
    T_tem = np.hstack((rot_tem, t1))
    T = np.vstack((T_tem, np.array([0, 0, 0, 1])))
    return T


def Yomi_parameters(T):
    T_tem = np.copy(T)

    #print('T_tem is', T_tem)
    Rx = np.arctan2(T_tem[2,1],T_tem[2,2])
    #if T[2,1]/np.sin(Rx) * T[2, 0] > 0:
    #Ry = np.arccos(T[2,1]/np.sin(Rx)) - math.pi
    Ry = np.arcsin(-T_tem[2,0])
    # if T_tem[2,0] > 0 and T[2,1]/np.sin(Rx) > 0:
    #     Ry = np.arcsin(T_tem[2,0])
    # elif T_tem[2,0] > 0 and T[2,1]/np.sin(Rx) < 0:
    #     Ry = np.arcsin(T_tem[2,0]) + math.pi
    # elif T_tem[2,0] < 0 and T[2,1]/np.sin(Rx) > 0:
    #     Ry = np.arcsin(T_tem[2, 0])
    # elif T_tem[2,0] < 0 and T[2,1]/np.sin(Rx) < 0:
    #     Ry = np.arcsin(T_tem[2,0]) + math.pi
    # else:
    #     Ry = np.arcsin(T_tem[2, 0])
    #print('siny',-T_tem[2,0] )
    #print('cosy', T[2,1]/np.sin(Rx))
    #print('Ry is', Ry)
    #Ry = np.arctan2((-T_tem[2,0]), (T[2,1]/np.sin(Rx)))
    Rz = np.arctan2(T[1,0],T[0,0])
    #print('angles are', Rx, Ry, Rz)
    A = np.array([[np.cos(Ry)*np.cos(Rz), np.cos(Rz)*np.sin(Rx)*np.sin(Ry)- np.cos(Rx)*np.sin(Rz), np.cos(Rx)*np.cos(Rz)*np.sin(Ry)+np.sin(Rx)*np.sin(Rz)],
                   [np.cos(Ry)*np.sin(Rz), np.cos(Rz)*np.cos(Rx)+ np.sin(Rz)*np.sin(Ry)*np.sin(Rx), -np.cos(Rz)*np.sin(Rx)+np.cos(Rx)*np.sin(Ry)*np.sin(Rz)],
                   [-np.sin(Ry), np.cos(Ry)*np.sin(Rx), np.cos(Rx)*np.cos(Ry)]])

    Tx = np.matmul(np.linalg.inv(A), T_tem[0:3,3])[0]
    Ty = np.matmul(np.linalg.inv(A), T_tem[0:3,3])[1]
    Tz = np.matmul(np.linalg.inv(A), T_tem[0:3,3])[2]

    Link = np.array([Tx, Ty, Tz, Rz, Ry, Rx])
    return Link


def convert_Yomi_convention(kdl,rot, t):
    rot_tem = np.copy(rot)
    t_tem = np.copy(t)
    kdl_tem = np.copy(kdl)
    T = convert_homo_transform(rot_tem, t_tem)
    T0 = Yomikin.Yomi_Base_Matrix(kdl_tem[0:6])
    #print('T is', T)
    #print('T newbase is', T0)
    T_newbase = np.matmul(T, T0)
    #print('T newbase is', T_newbase)
    Link0_new = Yomi_parameters(T_newbase)
    Tverify = Yomikin.Yomi_Base_Matrix(Link0_new)
    #print('T verify is', Tverify)
    kdl_new = np.insert(kdl_tem[6:], 0, Link0_new)
    return kdl_new


# For debugging

# Calculate three vectors: base to ee, J1 to ee, J2 to ee
def eeVector(kdl, joint_v):
    kdl_tem = np.copy(kdl)
    joint_test = np.copy(joint_v)
    mt, nt = np.shape(joint_test)
    count = np.zeros(mt)
    V1e = np.zeros((mt, 3))
    V2e = np.zeros((mt, 3))
    Vbe = np.zeros((mt, 3))
    for i in range(mt):
        joint_tem = np.insert(joint_test[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_tem)
        ee = np.matmul(pos[7], [0, 0, 0, 1]);
        ee = ee[0:3]
        j1 = np.matmul(pos[0], [0, 0, 0, 1]);
        j1 = j1[0:3]
        j2 = np.matmul(pos[1], [0, 0, 0, 1]);
        j2 = j2[0:3]

        V1e[i, :] = ee - j1
        V2e[i, :] = ee - j2
        Vbe[i, :] = ee
        count[i] = i
    return Vbe, V1e, V2e, count

def eeVector_rtip(kdl, joint_v, rtip):
    kdl_tem = np.copy(kdl)
    joint_test = np.copy(joint_v)
    rtip_tem = np.copy(rtip)
    rtip_homo = np.insert(rtip_tem,3,1)
    mt, nt = np.shape(joint_test)
    count = np.zeros(mt)
    V1e = np.zeros((mt, 3))
    V2e = np.zeros((mt, 3))
    Vbe = np.zeros((mt, 3))
    for i in range(mt):
        joint_tem = np.insert(joint_test[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_tem)
        #print('pos is', pos[7])
        #print('rtip_homo is', rtip_homo)
        ee = np.matmul(pos[7][0:3, 0:3], rtip_tem) + pos[7][0:3,3]
        #ee = np.matmul(pos[7], rtip_homo)
        #print('ee is', ee)
        #ee = np.matmul(pos[7], [0, 0, 0, 1]);
        ee = ee[0:3]
        j1 = np.matmul(pos[0], [0, 0, 0, 1]);
        j1 = j1[0:3]
        j2 = np.matmul(pos[1], [0, 0, 0, 1]);
        j2 = j2[0:3]

        V1e[i, :] = ee - j1
        V2e[i, :] = ee - j2
        Vbe[i, :] = ee
        count[i] = i
    return Vbe, V1e, V2e, count

# Calculate the differences bewteen two vectors: ve, va, vd
# ve is the error vector: ve = v1 - v2
# vd is the difference of the length of two vectors vd = length(v1)-length(v2)
# va is the angle between two vectors: va = angle between v1 and v2
def difVector(vec1, vec2):
    vec1_tem = np.copy(vec1)
    vec2_tem = np.copy(vec2)
    if np.shape(vec1_tem) != np.shape(vec2_tem):
        print('Error! Two vectors should be of the same size')
        print('Results are not correct')
        m = 10
    elif np.shape(vec1_tem) == np.shape(vec2_tem):
        m = np.shape(vec1_tem)[0]
    ve = np.zeros(m)
    vd = np.zeros(m)
    va = np.zeros(m)
    for i in range(m):
        ve_tem = vec1_tem[i, :] - vec2[i, :]
        ve[i] = np.sqrt(np.sum(ve_tem[0:3] ** 2)) * 1000
        vd[i] = np.sqrt(np.sum(vec1_tem[i, :] ** 2)) * 1000 - np.sqrt(np.sum(vec2_tem[i, :] ** 2)) * 1000
        va[i] = np.arccos(np.matmul(vec1_tem[i, :], vec2_tem[i, :]) / (
                    np.sqrt(np.sum(vec1_tem[i, :] ** 2)) * np.sqrt(np.sum(vec2_tem[i, :] ** 2))))
    return ve, vd, va

def Count_array(m):
    i = m
    count = np.zeros(i)
    for j in range(i):
        count[j] = j
    return count


# ---------------------------------------- Define Variables -------------------------------------- #
DEBUG_FLAG = 0
REMOVE_OUTLIER_BB = 1   # remove outliers in ball bar calibration
REMOVE_OUTLIER_PV = 1   # remove outliers in pivot calibration
kp_guess = np.transpose(np.array([0.0, -0.16, 0.0, 0.0, 0.0, -math.pi / 2,
                                  0.0, 0.0, 0.08, 0.0, math.pi / 2, 0.0,
                                  -0.26, 0.00, 0.0, 0.0, math.pi / 2, 0.0,
                                  0.0, 0.00, -0.05, 0.0, 0.0, math.pi / 2,
                                  0.0, 0.24, 0.0, 0.0, 0.0, -math.pi / 2,
                                  0.0, 0.0, -0.05, 0.0, 0.0, math.pi / 2,
                                  0.0, 0.03, 0.1, 0.0, 0.0, math.pi]))

kp_fit = np.transpose(np.array([1, 0, 0, 0, 0, 1,
                                1, 1, 0, 1, 1, 0,
                                1, 1, 0, 1, 1, 0,
                                1, 1, 0, 1, 0, 1,
                                1, 1, 0, 1, 0, 1,
                                1, 1, 0, 1, 0, 1,
                                1, 1, 1, 0, 0, 0]))

kp_guess_common = np.transpose(np.array([-0.08, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, -0.16, 0.0, 0.0, 0.0, -math.pi / 2,
                                         0.0, 0.0, 0.08, 0.0, math.pi / 2, 0.0,
                                         -0.26, 0.00, 0.0, 0.0, math.pi / 2, 0.0,
                                         0.0, 0.00, -0.05, 0.0, 0.0, math.pi / 2,
                                         0.0, 0.24, 0.0, 0.0, 0.0, -math.pi / 2,
                                         0.0, 0.0, -0.05, 0.0, 0.0, math.pi / 2,
                                         0.0, 0.03, 0.1, 0.0, 0.0, math.pi]))

kp_fit_common = np.transpose(np.array([1, 1, 1, 0, 0, 0,
                                       1, 0, 0, 0, 0, 1,  # Link1
                                       1, 1, 0, 1, 1, 0,
                                       1, 1, 0, 1, 1, 0,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 0, 0]))

kp_fit_common_full = np.transpose(np.array([1, 1, 1, 0, 0, 0,
                                       1, 0, 0, 0, 0, 1,  # Link1
                                       1, 1, 0, 1, 1, 0,
                                       1, 1, 0, 1, 1, 0,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 0, 0]))

# --------------------- Optimization Main Function ----------------------------------- #
#
kdl1 = opt_full(Data.joint_sb_B_3, Data.bb_fs, Data.joint_mb_B_3, Data.bb_fm, Data.joint_lb_B_3, Data.bb_fl, Data.joint_s1_B_3, Data.joint_s2_B_3, Data.joint_s3_B_3)
kdl2 = opt_full(Data.joint_sb_Xiao35_1, Data.bb_fs, Data.joint_mb_Xiao35_1, Data.bb_fm, Data.joint_lb_Xiao35_1, Data.bb_fl, Data.joint_s1_Xiao35_1, Data.joint_s2_Xiao35_1, Data.joint_s3_Xiao35_1)




# ------------------------------ Results verification in Working Volume----------------------------------------#
joint_test = Data.joint_poses
ee1, x21,x31,x41 = eeVector(kdl1,joint_test)
ee2, x2,x3,x4 = eeVector(kdl2,joint_test)

difee = np.sqrt(np.sum((ee1 - ee2) ** 2, axis=1))
countn = Count_array(np.shape(ee1)[0])

ve_ms13, vd_ms13, va_ms13 = difVector(ee1, ee2)
ve_mean = np.mean(ve_ms13)
ve_std = np.std(ve_ms13)
vd_mean = np.mean(vd_ms13)
vd_std = np.std(vd_ms13)
va_mean = np.mean(va_ms13)
va_std = np.std(va_ms13)

print('ve mean and std are', ve_mean, ve_std)
print('va mean and std are', va_mean, va_std)
print('vd mean and std are', vd_mean, vd_std)



# --------------------------------------------- Plotting -----------------------------------------------#
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(ee1[:, 0] * 1000, ee1[:, 1] * 1000, ee1[:, 2] * 1000, zdir='z', s=20, c='r',
           rasterized=True)
ax.scatter(ee2[:, 0] * 1000, ee2[:, 1] * 1000, ee2[:, 2] * 1000, zdir='z', s=20, c='g',
           rasterized=True)
plt.title('End effector scatter')
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')


fig2 = plt.figure()
plt.scatter(countn, difee * 1000, alpha=0.5, c='r', label='m1-s3')
plt.legend()
plt.title('ee errors')
plt.xlabel('Trial Number')
plt.ylabel('Errors (mm)')


fig3 = plt.figure()
plt.scatter(countn, ve_ms13, alpha=0.5, c='c', label='m1-s3')
plt.legend()
plt.title('Vector error (new base to ee)')
plt.xlabel('Trial Number')
plt.ylabel('Length error of vector (mm)')



fig4 = plt.figure()
plt.scatter(countn, va_ms13, alpha=0.5, c='b', label='fm3')
plt.legend()
plt.title('Vector angle comparison (new base to ee)')
plt.xlabel('Trial Number')
plt.ylabel('Angle of vector (rad)')


fig5 = plt.figure()
plt.scatter(countn, vd_ms13, alpha=0.5, c='b', label='fm3')
plt.legend()
plt.title('Vector length comparison (new base to ee)')
plt.xlabel('Trial Number')
plt.ylabel('Difference of length of vectors (mm)')


# Plot histogram, info:
# https://stackoverflow.com/questions/38650550/cant-get-y-axis-on-matplotlib-histogram-to-display-probabilities
fig6 = plt.figure()
plt.hist(ve_ms13, weights=np.ones_like(ve_ms13) / len(ve_ms13), facecolor = 'g', alpha = 0.75)
plt.xlabel('Tip Position Errors (mm)')
plt.ylabel('Probability')
plt.title('Histogram of Tip position errors')
#plt.text(60, .025, r'$\mu=$', ve_mean,'$\sigma=$', ve_std)
plt.xlim(0, 1)
plt.ylim(0, 0.35)
plt.grid(True)

fig7 = plt.figure()
plt.hist(va_ms13, weights=np.ones_like(va_ms13) / len(va_ms13), facecolor = 'g', alpha = 0.75)
plt.xlabel('Tip Position Angular Errors (rad)')
plt.ylabel('Probability')
plt.title('Histogram of Tip position angular errors')
#plt.text(60, .025, r'$\mu=$', ve_mean,'$\sigma=$', ve_std)
plt.xlim(0, 0.004)
plt.ylim(0, 0.35)
plt.grid(True)

fig8 = plt.figure()
plt.hist(vd_ms13, weights=np.ones_like(vd_ms13) / len(vd_ms13), facecolor = 'g', alpha = 0.75)
plt.xlabel('Vector Length Errors (mm)')
plt.ylabel('Probability')
plt.title('Histogram of vector length errors')
#plt.text(60, .025, r'$\mu=$', ve_mean,'$\sigma=$', ve_std)
#plt.xlim(0, 0.1)
#plt.ylim(0, 0.03)
plt.grid(True)

#
#
#
# fig9 = plt.figure()
# plt.hist(bar_est1, weights=np.ones_like(bar_est1) / len(bar_est1), facecolor = 'g', alpha = 0.75)
# plt.xlabel('Bar 1 Length (mm)')
# plt.ylabel('Probability')
# plt.title('Histogram of Bar Length')
# #plt.text(60, .025, r'$\mu=$', ve_mean,'$\sigma=$', ve_std)
# #plt.xlim(0, 1)
# #plt.ylim(0, 0.35)
# plt.grid(True)
#
# fig10 = plt.figure()
# plt.hist(bar_est2, weights=np.ones_like(bar_est2) / len(bar_est2), facecolor = 'g', alpha = 0.75)
# plt.xlabel(' Bar 2 Length (mm)')
# plt.ylabel('Probability')
# plt.title('Histogram of Bar Length')
# #plt.text(60, .025, r'$\mu=$', ve_mean,'$\sigma=$', ve_std)
# #plt.xlim(0, 1)
# #plt.ylim(0, 0.35)
# plt.grid(True)


plt.show()

