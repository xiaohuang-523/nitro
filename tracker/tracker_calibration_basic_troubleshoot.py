import Readers as Yomiread
import Kinematics as Yomikin
import numpy as np
import timeit
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import math
import os
#import plot
import Writers as Yomiwrite
import compare as YomiComp

# Delete after verification
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import datetime


def svdsolve(a,b):
    u, s, v = np.linalg.svd(a)
    c = np.dot(u.T, b)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(v.T, w)
    return x


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

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


# duplicate 1st arg based on measurements. Prepare data for optimization function
def dup_bl(bl, measurements):
    m, n = np.shape(measurements)
    return np.array(np.repeat(bl, m))

#Estimate common base sphere locations for general bar method
def est_sphere_CB(joint):
    joint_tem = joint
    m, n = np.shape(joint_tem)
    EE_p_est_x = []
    EE_p_est_y = []
    EE_p_est_z = []
    for k in range(m):
        EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(kp_guess_common, joint_tem[k, :])
        # EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(tracker_kdl, joint_tem[k, :])
        EE_p_est_tem = np.matmul(EE_T_est_tem[7], [0, 0, 0, 1])
        # EE_p_est_tem = np.matmul(EE_T_est_tem[6], [0, 0, 0, 1])
        EE_p_est_x.append(EE_p_est_tem[0])
        EE_p_est_y.append(EE_p_est_tem[1])
        EE_p_est_z.append(EE_p_est_tem[2])
    ball = fitSphere(np.array(EE_p_est_x), np.array(EE_p_est_y), np.array(EE_p_est_z))[0:3]
    return np.array(ball)

def solve_commonbase(divot1, divot2, divot3):
    s1 = np.asarray(divot1)
    s2 = np.asarray(divot2)
    s3 = np.asarray(divot3)
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

def AOS_pivot_calibration(joint, kdl, count):
    A = np.zeros((3 * count, 6))
    b = np.zeros(3 * count)
    for i in range(count):
        joint_test = np.insert(joint[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl, joint_test)
        t_tem = np.matmul(pos[7], [0, 0, 0, 1])
        R_tem = pos[7]
        A[3 * i, :] = np.append(R_tem[0, 0:3], -np.array([1, 0, 0]))
        A[3 * i + 1, :] = np.append(R_tem[1, 0:3], -np.array([0, 1, 0]))
        A[3 * i + 2, :] = np.append(R_tem[2, 0:3], -np.array([0, 0, 1]))
        b[3 * i] = t_tem[0]
        b[3 * i + 1] = t_tem[1]
        b[3 * i + 2] = t_tem[2]
    t = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), -b)
    cond_AOS = np.linalg.cond(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)))
    # information from https://sukhbinder.wordpress.com/2013/03/26/solving-axb-by-svd/
    #t_lu = np.linalg.solve(A,-b)
    #t_svd = svdsolve(A, -b)
    print('t is', t)
    #print('t_svd is', t_svd)
    #print('t_lu is', t_lu)
    return t[0:3], t[3:6], cond_AOS

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
        ee = np.matmul(pos[7], [0, 0, 0, 1])
        ee = ee[0:3]
        j1 = np.matmul(pos[0], [0, 0, 0, 1])
        j1 = j1[0:3]
        j2 = np.matmul(pos[1], [0, 0, 0, 1])
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
        j1 = np.matmul(pos[0], [0, 0, 0, 1])
        j1 = j1[0:3]
        j2 = np.matmul(pos[1], [0, 0, 0, 1])
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

def looperror(cal, jointangles, bb, ball, count):
    kp_tem = np.copy(kp_guess_common)
    cal_tem = np.copy(cal)
    kp_fit_tem = np.copy(kp_fit_common)

    cal_tem = unpack(kp_tem, cal_tem, kp_fit_tem)
    err = np.zeros(shape=(count,))
    for x in range(count):
        joint_tem_f = np.insert(jointangles[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(cal_tem, joint_tem_f)
        # If the common reference is removed, use '6'
        # If the common reference is kept, use '7'
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball) - bb[x]

    print('Ball bar calibration rms is', np.sqrt(np.sum(err ** 2, axis=0)/count)*1000, 'mm')
    err_mean_yomi = np.mean(err)
    std_yomi = np.sqrt(np.sum((err-err_mean_yomi)**2)/count)*1000
    print('Yomi RMS is', std_yomi)
    print('Yomi variance is',np.sum((err-err_mean_yomi)**2)/count)
    print('np variance is', np.var(err))
    print('Mean residual distance error is', np.mean(err)*1000, 'mm')
    print('Standard deviation of residual distance is', np.std(err, axis=0)*1000, 'mm')
    return err


def looperror_debug(cal, jointangles, bb, ball, count):
    kp_tem = np.copy(kp_guess_common)
    cal_tem = np.copy(cal)
    kp_fit_tem = np.copy(kp_fit_common)

    cal_tem = unpack(kp_tem, cal_tem, kp_fit_tem)
    err = np.zeros(shape=(count,))
    for x in range(count):
        joint_tem_f = np.insert(jointangles[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(cal_tem, joint_tem_f)
        # If the common reference is removed, use '6'
        # If the common reference is kept, use '7'
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball) - bb[x]

    #print('Ball bar calibration rms is', np.sqrt(np.sum(err ** 2, axis=0)/count)*1000, 'mm')
    err_mean_yomi = np.mean(err)
    std_yomi = np.sqrt(np.sum((err-err_mean_yomi)**2)/count)*1000
    #print('Yomi RMS is', std_yomi)
    #print('Yomi variance is',np.sum((err-err_mean_yomi)**2)/count)
    #print('np variance is', np.var(err))
    #print('Mean residual distance error is', np.mean(err)*1000, 'mm')
    #print('Standard deviation of residual distance is', np.std(err, axis=0)*1000, 'mm')
    return err


def opt_bb(joint, bb, ball, count):
    kp_guess_common_tem = np.copy(kp_guess_common)
    kp_fit_common_tem = np.copy(kp_fit_common)
    opt = pack(kp_guess_common_tem, kp_fit_common_tem)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror, opt,
                                                         (joint, bb, ball, count),
                                                         ftol=1e-10, xtol=1e-7,
                                                         full_output=True)
    print('integer flag is', ier)
    print('message is', mesg)
    jac = infodict["fjac"]
    jacobian_condition_number = np.linalg.cond(jac)
    print('Ball bar calibration condition number is', jacobian_condition_number)
    tem_p = p[:]
    kdl = tem_p[:]
    print('kdl is', kdl)
    kdl_unpack = unpack(kp_guess_common_tem, kdl, kp_fit_common_tem)
    print('kdl_unpack', kdl_unpack)
    return kdl_unpack


def opt_bb_debug(joint, bb, ball, count):
    print('number of data points is', count)
    kp_guess_common_tem = np.copy(kp_guess_common)
    kp_fit_common_tem = np.copy(kp_fit_common)
    opt = pack(kp_guess_common_tem, kp_fit_common_tem)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror_debug, opt,
                                                         (joint, bb, ball, count),
                                                         ftol=1e-10, xtol=1e-7,
                                                         full_output=True)
    jac = infodict["fjac"]
    jacobian_condition_number = np.linalg.cond(jac)
    print('Ball bar calibration condition number is', jacobian_condition_number)
    tem_p = p[:]
    kdl = tem_p[:]
    kdl_unpack = unpack(kp_guess_common_tem, kdl, kp_fit_common_tem)
    s1t_no, s1b_no, cond = AOS_pivot_calibration(joint, kdl_unpack, joint.shape[0])
    rms, mean, std, countx = est_rms_pivot(joint,kdl_unpack,s1t_no,s1b_no,joint.shape[0])


    return kdl_unpack, rms, mean, std


def create_bb_length_map(bb1_angle_count, bb2_angle_count, bb3_angle_count, b1_length, b2_length, b3_length):
    bb_count_length = {}
    bb_count_length[bb1_angle_count] = b1_length
    bb_count_length[bb2_angle_count] = b2_length
    bb_count_length[bb3_angle_count] = b3_length
    return bb_count_length


def create_bb_length_map_2bar(bb1_angle_count, bb2_angle_count, b1_length, b2_length):
    bb_count_length = {}
    bb_count_length[bb1_angle_count] = b1_length
    bb_count_length[bb2_angle_count] = b2_length
    return bb_count_length


def opt_full(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3, joint_s1, joint_s2, joint_s3):
    print('Optimizing with single ball')
    bb_count_length = create_bb_length_map(joint_bb1.shape[0], joint_bb2.shape[0], joint_bb3.shape[0], bb_1, bb_2, bb_3)

    bb1 = dup_bl(bb_1, joint_bb1)
    sphere1 = est_sphere_CB(joint_bb1)

    bb2 = dup_bl(bb_2, joint_bb2)
    sphere2 = est_sphere_CB(joint_bb2)

    bb3 = dup_bl(bb_3, joint_bb3)
    sphere3 = est_sphere_CB(joint_bb3)

    all_joint_angles = np.vstack((joint_bb1, joint_bb2, joint_bb3))

    ball_bar_values = np.copy(bb1)
    ball_bar_values = np.append(ball_bar_values, bb2)
    ball_bar_values = np.append(ball_bar_values, bb3)
    ball = (sphere1 + sphere2 + sphere3) / 3

    # Solve bb calibration
    time1 = np.datetime64('now')
    kdl_tem = opt_bb(all_joint_angles, ball_bar_values, ball, all_joint_angles.shape[0])
    time2 = np.datetime64('now')
    print('calculation time is', time2-time1)
    print('kdl_tem is', kdl_tem)

    # Remove outliers in bb calibration
    err_data, out_indx, rms_data = find_outlier_bb(all_joint_angles, ball_bar_values, ball, kdl_tem, all_joint_angles.shape[0])
    j_2 = remove_outlier(all_joint_angles, out_indx)
    bb_2 = remove_outlier(ball_bar_values, out_indx)
    print('j size is', np.shape(all_joint_angles))
    print('j size after delete is', np.shape(j_2))

    inlier_count = j_2.shape[0]
    # Solve bb calibration after outlier rejection
    kdl_tem = opt_bb(j_2, bb_2, ball, inlier_count)

    # solve pivot calibrations
    s1_count = joint_s1.shape[0]
    s2_count = joint_s2.shape[0]
    s3_count = joint_s3.shape[0]

    s1t, s1b, cond1 = AOS_pivot_calibration(joint_s1, kdl_tem, s1_count)
    s2t, s2b, cond2 = AOS_pivot_calibration(joint_s2, kdl_tem, s2_count)
    s3t, s3b, cond3 = AOS_pivot_calibration(joint_s3, kdl_tem, s3_count)
    print('s1t is', s1t)
    print('s1b is', s1b)

    # remove outliers in pivoting
    s1t_2, s1b_2, rms_s1, cond_s1, mean_s1, std_s1, count_s1 = remove_outlier_pivot(joint_s1, kdl_tem, s1t, s1b, s1_count)
    s2t_2, s2b_2, rms_s2, cond_s2, mean_s2, std_s2, count_s2 = remove_outlier_pivot(joint_s2, kdl_tem, s2t, s2b, s2_count)
    s3t_2, s3b_2, rms_s3, cond_s3, mean_s3, std_s3, count_s3 = remove_outlier_pivot(joint_s3, kdl_tem, s3t, s3b, s3_count)

    print('s1 pivot calibration condition number is', cond_s1)
    print('s1 pivot calibration rms is', rms_s1*1000, 'mm')
    print('s1 mean is', mean_s1*1000, 'mm')
    print('s1 std is', std_s1*1000, 'mm')
    print('s1 count is', count_s1)
    print('s2 pivot calibration condition number is', cond_s2)
    print('s2 pivot calibration rms is', rms_s2*1000, 'mm')
    print('s2 mean is', mean_s2*1000, 'mm')
    print('s2 std is', std_s2*1000, 'mm')
    print('s2 count is', count_s2)
    print('s3 pivot calibration condition number is', cond_s3)
    print('s3 pivot calibration rms is', rms_s3*1000, 'mm')
    print('s3 mean is', mean_s3*1000, 'mm')
    print('s3 std is', std_s3*1000, 'mm')
    print('s3 count is', count_s3)

    print('s1 is', s1b_2)
    print('s2 is', s2b_2)
    print('s3 is', s3b_2)
    # solve common base transformation
    rot, t = solve_commonbase(s1b_2, s2b_2, s3b_2)
    print('rot is', rot)
    print('t is', t)

    print('old ball position is', ball)
    ball_tem = np.insert(ball, 3, 1.0)
    ball_new = np.matmul(convert_homo_transform(rot, t), ball_tem)
    print('new ball position is', ball_new)
    s1_tem = np.insert(s1b_2, 3, 1.0)
    print('old s1 is', s1b_2)
    print('new s1 is', np.matmul(convert_homo_transform(rot, t), s1_tem))

    kdl_final = convert_Yomi_convention(kdl_tem,rot,t)

    comfirm_rms(kdl_final, j_2, bb_2, ball_new[0:3], inlier_count)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists("C:\\Calibration\\tracker_kdl\\"):
        os.makedirs("C:\\Calibration\\tracker_kdl\\")
    filename1 = "C:\\Calibration\\tracker_kdl\\tracker_kdl_" + time_stamp + ".txt"
    np.savetxt(filename1, kdl_final, newline=",\n")
    return kdl_final

def find_outlier_bb(joint, bb, ball, kdl, joint_angle_count):
    err = np.zeros(shape=(joint_angle_count,))
    out_indx = np.array([])
    for x in range(joint_angle_count):
        joint_tem_f = np.insert(joint[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl, joint_tem_f)
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball) - bb[x]
        if np.abs(err[x,]) > 0.0001:
            out_indx = np.append(out_indx, np.array([x]))
    eer_rms = np.var(err)
    return err, out_indx, eer_rms

def find_outlier_pivot(joint, kdl, rtip, divot, joint_angle_count):
    rtip_homo = np.insert(rtip,3,1)
    err = np.zeros(shape=(joint_angle_count,))
    out_indx = np.array([])
    for x in range(joint_angle_count):
        joint_tem_f = np.insert(joint[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl, joint_tem_f)
        eepos = np.matmul(eepos[7], rtip_homo)
        err[x,] = np.linalg.norm(eepos[0:3]-divot)
    for x in range(joint_angle_count):
        if err[x,] > 0.00015:
            out_indx = np.append(out_indx, np.array([x]))
    print('divot 0 is', eepos[0:3])
    return err, out_indx

def est_rms_pivot(joint, kdl, rtip, divot, joint_angle_count):
    rtip_homo = np.insert(rtip,3,1)
    err = np.zeros(shape=(joint_angle_count,))
    for x in range(joint_angle_count):
        joint_tem_f = np.insert(joint[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl, joint_tem_f)
        eepos = np.matmul(eepos[7], rtip_homo)
        err[x,] = np.linalg.norm(eepos[0:3]-divot)
    eer_rms = np.sqrt(np.sum(err ** 2, axis=0) /joint_angle_count)
    eer_mean = np.sum(err, axis=0)/ joint_angle_count
    eer_std = np.std(err, axis=0)
    err_mean_yomi = np.mean(err)
    rms_yomi = np.sqrt(np.sum((err-err_mean_yomi)**2)/joint_angle_count)

    return rms_yomi, eer_mean, eer_std, joint_angle_count

def remove_outlier(array, out_indx):
    array_tem = np.copy(array)
    for index in sorted(out_indx, reverse=True):
        array_tem = np.delete(array_tem, int(index), axis=0)
    return array_tem

def remove_outlier_pivot(joint, kdl, hole, divot, count):
    err_data, out_indx = find_outlier_pivot(joint, kdl, hole, divot, count)
    print(np.shape(out_indx)[0], 'out of', np.shape(joint)[0], 'outliers were removed')
    joint_s1_no = remove_outlier(joint, out_indx)
    s1t_no, s1b_no, cond = AOS_pivot_calibration(joint_s1_no, kdl, joint_s1_no.shape[0])
    rms, mean, std, countx = est_rms_pivot(joint_s1_no,kdl,s1t_no,s1b_no,joint_s1_no.shape[0])
    return s1t_no, s1b_no, rms, cond, mean, std, countx

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
    Rx = np.arctan2(T_tem[2,1],T_tem[2,2])
    Ry = np.arcsin(-T_tem[2,0])
    Rz = np.arctan2(T[1,0],T[0,0])
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
    print('T newbase is', T_newbase)
    Link0_new = Yomi_parameters(T_newbase)
    Tverify = Yomikin.Yomi_Base_Matrix(Link0_new)
    #print('T verify is', Tverify)
    kdl_new = np.insert(kdl_tem[6:], 0, Link0_new)
    return kdl_new

def convert_Yomi_convention_matrix(kdl,tf):
    kdl_tem = np.copy(kdl)
    T = np.copy(tf)
    T0 = Yomikin.Yomi_Base_Matrix(kdl_tem[0:6])
    #print('T is', T)
    #print('T newbase is', T0)
    T_newbase = np.matmul(T, T0)
    print('T newbase is', T_newbase)
    Link0_new = Yomi_parameters(T_newbase)
    Tverify = Yomikin.Yomi_Base_Matrix(Link0_new)
    #print('T verify is', Tverify)
    kdl_new = np.insert(kdl_tem[6:], 0, Link0_new)
    return kdl_new

def opt_debug(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3):
    print('Optimizing with single ball')
    bb_count_length = create_bb_length_map(joint_bb1.shape[0], joint_bb2.shape[0], joint_bb3.shape[0], bb_1, bb_2, bb_3)

    bb1 = dup_bl(bb_1, joint_bb1)
    sphere1 = est_sphere_CB(joint_bb1)

    bb2 = dup_bl(bb_2, joint_bb2)
    sphere2 = est_sphere_CB(joint_bb2)

    bb3 = dup_bl(bb_3, joint_bb3)
    sphere3 = est_sphere_CB(joint_bb3)

    all_joint_angles = np.vstack((joint_bb1, joint_bb2, joint_bb3))

    ball_bar_values = np.copy(bb1)
    ball_bar_values = np.append(ball_bar_values, bb2)
    ball_bar_values = np.append(ball_bar_values, bb3)
    ball = (sphere1 + sphere2 + sphere3) / 3

    # Solve bb calibration
    time1 = np.datetime64('now')
    kdl_tem = opt_bb(all_joint_angles, ball_bar_values, ball, all_joint_angles.shape[0])
    time2 = np.datetime64('now')
    print('calculation time is', time2-time1)

    # For debugging, read orignal kdl without outlier rejection
    kdl_no_rejection = kdl_tem

    # Remove outliers in bb calibration
    err_data, out_indx, rms_data = find_outlier_bb(all_joint_angles, ball_bar_values, ball, kdl_tem, all_joint_angles.shape[0])
    j_2 = remove_outlier(all_joint_angles, out_indx)
    bb_2 = remove_outlier(ball_bar_values, out_indx)
    print('j size is', np.shape(all_joint_angles))
    print('j size after delete is', np.shape(j_2))

    inlier_count = j_2.shape[0]
    # Solve bb calibration after outlier rejection
    kdl_tem = opt_bb(j_2, bb_2, ball, inlier_count)
    return kdl_tem, kdl_no_rejection

def opt_debug_2bar_test(joint_bb1, bb_1, joint_bb2, bb_2):
    print('Optimizing with single ball')
    bb_count_length = create_bb_length_map_2bar(joint_bb1.shape[0], joint_bb2.shape[0], bb_1, bb_2)

    bb1 = dup_bl(bb_1, joint_bb1)
    sphere1 = est_sphere_CB(joint_bb1)

    bb2 = dup_bl(bb_2, joint_bb2)
    sphere2 = est_sphere_CB(joint_bb2)


    all_joint_angles = np.vstack((joint_bb1, joint_bb2))

    ball_bar_values = np.copy(bb1)
    ball_bar_values = np.append(ball_bar_values, bb2)
    ball = (sphere1 + sphere2) / 2

    # Solve bb calibration
    time1 = np.datetime64('now')
    kdl_tem = opt_bb(all_joint_angles, ball_bar_values, ball, all_joint_angles.shape[0])
    time2 = np.datetime64('now')
    print('calculation time is', time2-time1)

    # For debugging, read orignal kdl without outlier rejection
    kdl_no_rejection = kdl_tem

    # Remove outliers in bb calibration
    err_data, out_indx, rms_data = find_outlier_bb(all_joint_angles, ball_bar_values, ball, kdl_tem, all_joint_angles.shape[0])
    j_2 = remove_outlier(all_joint_angles, out_indx)
    bb_2 = remove_outlier(ball_bar_values, out_indx)
    print('j size is', np.shape(all_joint_angles))
    print('j size after delete is', np.shape(j_2))

    inlier_count = j_2.shape[0]
    # Solve bb calibration after outlier rejection
    kdl_tem = opt_bb(j_2, bb_2, ball, inlier_count)
    return kdl_tem

def opt_bb_debug_single_ball(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3):
    print('Debugging single ball')
    bb_count_length = create_bb_length_map(joint_bb1.shape[0], joint_bb2.shape[0], joint_bb3.shape[0], bb_1, bb_2, bb_3)

    bb1 = dup_bl(bb_1, joint_bb1)
    sphere1 = est_sphere_CB(joint_bb1)

    bb2 = dup_bl(bb_2, joint_bb2)
    sphere2 = est_sphere_CB(joint_bb2)

    bb3 = dup_bl(bb_3, joint_bb3)
    sphere3 = est_sphere_CB(joint_bb3)

    #all_joint_angles = np.vstack((joint_bb1, joint_bb2, joint_bb3))

    #ball_bar_values = np.copy(bb1)
    #ball_bar_values = np.append(ball_bar_values, bb2)
    #ball_bar_values = np.append(ball_bar_values, bb3)
    #ball = (sphere1 + sphere2 + sphere3) / 3

    # Solve bb calibration
    kdl_1, rms1, mean1, std1 = opt_bb_debug(joint_bb1, bb1, sphere1, joint_bb1.shape[0])
    print('short bar rms and mean is', rms1, mean1)

    kdl_2, rms2, mean2, std2 = opt_bb_debug(joint_bb2, bb2, sphere2, joint_bb2.shape[0])
    print('medium bar rms and mean is', rms2, mean2)

    kdl_3, rms3, mean3, std3 = opt_bb_debug(joint_bb3, bb3, sphere3, joint_bb3.shape[0])
    print('long bar rms and mean is', rms3, mean3)

    # Remove outliers in bb calibration
    #err_data1, out_indx1, rms_data1 = find_outlier_bb(joint_bb1, bb1, sphere1, kdl_1, joint_bb1.shape[0])
    #j_2 = remove_outlier(joint_bb1, out_indx1)
    #bb_2 = remove_outlier(bb1, out_indx1)
    #print('j size is', np.shape(joint_bb1))
    #print('j size after delete is', np.shape(j_2))

    #inlier_count = j_2.shape[0]
    # Solve bb calibration after outlier rejection
    #kdl_tem = opt_bb(j_2, bb_2, sphere1, inlier_count)

    kdl_1, kdl_1_no_outlier, out_indx_1 = outlier_debug(joint_bb1, bb1, sphere1, kdl_1)
    kdl_2, kdl_2_no_outlier, out_indx_2 = outlier_debug(joint_bb2, bb2, sphere2, kdl_2)
    kdl_3, kdl_3_no_outlier, out_indx_3 = outlier_debug(joint_bb3, bb3, sphere3, kdl_3)

    data1 = YomiComp.comp_kdl(kdl_1, kdl_1_no_outlier, PT_LIST)
    data2 = YomiComp.comp_kdl(kdl_2, kdl_2_no_outlier, PT_LIST)
    data3 = YomiComp.comp_kdl(kdl_3, kdl_3_no_outlier, PT_LIST)
    print('data1 is', data1)
    print('data2 is', data2)
    print('data3 is', data3)

    #fig = plt.figure()
        #plt.scatter(joint_bb1[int(i),0], joint_bb1[int(i),1], label = 'short bar outlier')
    for k in range(7):
        fig_tem = plt.figure()
        plt.scatter(range(joint_bb1.shape[0]), joint_bb1[:, k], c='blue')
        #count = 0
        for i in out_indx_1:
            #count += 1
            plt.scatter(int(i), joint_bb1[int(i), k], c = 'r')

        plt.title('sb joint ' + np.str(k))

    for k in range(7):
        fig_tem = plt.figure()
        plt.scatter(range(joint_bb2.shape[0]), joint_bb2[:, k], c='blue')
        #count = 0
        for i in out_indx_2:
            #count += 1
            plt.scatter(int(i), joint_bb2[int(i), k], c = 'r')

        plt.title('mb joint ' + np.str(k))

    for k in range(7):
        fig_tem = plt.figure()
        plt.scatter(range(joint_bb3.shape[0]), joint_bb3[:, k], c='blue')
        #count = 0
        for i in out_indx_3:
            #count += 1
            plt.scatter(int(i), joint_bb3[int(i), k], c = 'r')

        plt.title('lb joint ' + np.str(k))


    #for i in out_indx_2:
    #    plt.scatter(joint_bb2[i,0], joint_bb2[i,1], label = 'medium bar outlier')
    #for i in out_indx_3:
     #   plt.scatter(joint_bb3[i, 0], joint_bb3[i, 1], label='long bar outlier')





    #plt.legend()
    plt.show()

    return kdl_1

    #return kdl_1, kdl_2, kdl_3, kdl_1_no_outlier, kdl_2_no_outlier, kdl_3_no_outlier

def outlier_debug(joint_bb, bb_length, ball_center, kdl):
    err_data1, out_indx1, rms_data1 = find_outlier_bb(joint_bb, bb_length, ball_center, kdl, joint_bb.shape[0])
    j_2 = remove_outlier(joint_bb, out_indx1)
    bb_2 = remove_outlier(bb_length, out_indx1)
    print('j size is', np.shape(joint_bb))
    print('j size after delete is', np.shape(j_2))

    inlier_count = j_2.shape[0]
    # Solve bb calibration after outlier rejection
    kdl_tem, rms, mean, std = opt_bb_debug(j_2, bb_2, ball_center, inlier_count)


    return kdl, kdl_tem, out_indx1

def comfirm_rms(kdl, jointangles, bb, ball, count):
    err = np.zeros(shape=(count,))
    for x in range(count):
        joint_tem_f = np.insert(jointangles[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(kdl, joint_tem_f)
        # If the common reference is removed, use '6'
        # If the common reference is kept, use '7'
        eepos = np.matmul(eepos[7], [0, 0, 0, 1])
        err[x,] = np.linalg.norm(eepos[0:3] - ball) - bb[x]

    print('Ball bar calibration rms after base registration is', np.sqrt(np.sum(err ** 2, axis=0) / count) * 1000, 'mm')
    err_mean_yomi = np.mean(err)
    std_yomi = np.sqrt(np.sum((err - err_mean_yomi) ** 2) / count) * 1000
    print('after base registration Yomi RMS is', std_yomi)
    print('after base registration Yomi variance is', np.sum((err - err_mean_yomi) ** 2) / count)
    print('after base registration np variance is', np.var(err))
    print('after base registration Mean residual distance error is', np.mean(err) * 1000, 'mm')
    print('after base registration Standard deviation of residual distance is', np.std(err, axis=0) * 1000, 'mm')

# ---------------------------------------- Define Variables -------------------------------------- #
#kp_guess = np.transpose(np.array([0.0, -0.13, 0.0, 0.0, 0.0, -math.pi / 2,
#                                  0.0, 0.0, 0.08, 0.0, math.pi / 2, 0.0,
#                                  -0.26, 0.00, 0.0, 0.0, math.pi / 2, 0.0,
#                                  0.0, 0.00, -0.05, 0.0, 0.0, math.pi / 2,
#                                  0.0, 0.24, 0.0, 0.0, 0.0, -math.pi / 2,
#                                  0.0, 0.0, -0.05, 0.0, 0.0, math.pi / 2,
#                                  0.0, 0.03, 0.1, 0.0, 0.0, math.pi]))

# tracker_neo
#kp_guess = np.transpose(np.array([  0, -0.157, 0, 0, 0, -1.57079632679,
#                                    0, 0, 0.08, 0, 1.57079632679, 0,
#                                    -0.2, 0, 0, 0, 1.57079632679, 0,
#                                    -0.018, 0, -0.05, 0, 0, 1.57079632679,
#                                    0, 0.16, 0, 0, 0, -1.57079632679,
#                                    0, 0, -0.05, 0, 0,1.57079632679,
#                                    0.01, 0.038, 0.103, 0, 0, 3.141592653589793]))

#kp_fit = np.transpose(np.array([1, 0, 0, 0, 0, 1,
#                                 1, 1, 0, 1, 1, 0,
#                                 1, 1, 0, 1, 1, 0,
#                                 1, 1, 0, 1, 0, 1,
#                                 1, 1, 0, 1, 0, 1,
#                                 1, 1, 0, 1, 0, 1,
#                                 1, 1, 1, 0, 0, 0]))
#
# Original Initial Guess
#kp_guess_common = np.transpose(np.array([-0.08, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                          0.0, -0.13, 0.0, 0.0, 0.0, -math.pi / 2,
#                                          0.0, 0.0, 0.08, 0.0, math.pi / 2, 0.0,
#                                          -0.26, 0.00, 0.0, 0.0, math.pi / 2, 0.0,
#                                          0.0, 0.00, -0.05, 0.0, 0.0, math.pi / 2,
#                                          0.0, 0.24, 0.0, 0.0, 0.0, -math.pi / 2,
#                                          0.0, 0.0, -0.05, 0.0, 0.0, math.pi / 2,
#                                          0.0, 0.03, 0.1, 0.0, 0.0, math.pi]))
# Initial guess on Yomisettings
#kp_guess_common = np.transpose(np.array([-0.08, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                          0.0, -0.13, 0.0, 0.0, 0.0, -math.pi / 2,
#                                          0.0, 0.0, 0.08, 0.0, math.pi / 2, 0.0,
#                                          -0.26, 0.00, 0.0, 0.0, math.pi / 2, 0.0,
#                                          0.0, 0.00, -0.05, 0.0, 0.0, math.pi / 2,
#                                          0.0, 0.24, 0.0, 0.0, 0.0, -math.pi / 2,
#                                          0.0, 0.0, -0.05, 0.0, 0.0, math.pi / 2,
#                                          -0.03, 0.0, 0.1, 0.0, 0.0, math.pi]))

# tracker_neo
kp_guess_common = np.transpose(np.array([-0.07, 0, 0, 0, 0, 0,
                                    0, -0.157, 0, 0, 0, -1.57079632679,
                                    0, 0, 0.08, 0, 1.57079632679, 0,
                                    -0.2, 0, 0, 0, 1.57079632679, 0,
                                    -0.018, 0, -0.05, 0, 0, 1.57079632679,
                                    0, 0.16, 0, 0, 0, -1.57079632679,
                                    0, 0, -0.05, 0, 0,1.57079632679,
                                    0.01, 0.038, 0.103, 0, 0, 3.141592653589793]))

#kp_guess_common = np.transpose(np.array([-7.075799405416696719e-02,3.386418462633189550e-03,-1.703122920766883476e-04,5.719307725124044700e-02,1.072838093432388349e-03,-7.591087549637656404e-05,
#                                    0, -0.157, 0, 0, 0, -1.57079632679,
#                                    0, 0, 0.08, 0, 1.57079632679, 0,
#                                    -0.2, 0, 0, 0, 1.57079632679, 0,
#                                    -0.018, 0, -0.05, 0, 0, 1.57079632679,
#                                    0, 0.16, 0, 0, 0, -1.57079632679,
#                                    0, 0, -0.05, 0, 0,1.57079632679,
#                                    0.01, 0.038, 0.103, 0, 0, 3.141592653589793]))

#kp_guess_common = np.transpose(np.array([-0.07083587,
#-0.00066378,
#-0.00017031,
#0.00000000,
#0.00106674,
#-0.00013711,
#0.00019842,
#-0.15700000,
#0.00000000,
#0.05719304,
#0.00000000,
#-1.56692254,
#                                    0, 0, 0.08, 0, 1.57079632679, 0,
#                                    -0.2, 0, 0, 0, 1.57079632679, 0,
#                                    -0.018, 0, -0.05, 0, 0, 1.57079632679,
#                                    0, 0.16, 0, 0, 0, -1.57079632679,
#                                    0, 0, -0.05, 0, 0,1.57079632679,
#                                    0.01, 0.038, 0.103, 0, 0, 3.141592653589793]))

kp_fit_common = np.transpose(np.array([1, 1, 1, 0, 0, 0,  # Common link
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

#base = "G:\\My Drive\\Project\\Single-ball PT Calibration-revisit\\Testing Data\\"
#base = "G:\\My Drive\\Project\\Field Service Support\\Service Laptop-20201121T223452Z-001\\Service Laptop\\Output\\tracker-kincal-data-archive-20201121-123237-794"
#base = "G:\\My Drive\\Project\\Field Service Support\\Y-036\\Y-036 Another Run\\tracker-kincal-data"

base = "G:\\My Drive\\Project\\Field Service Support\\New_encoder\\Tracker_Daniel"
#base = "G:\\My Drive\\Project\\Field Service Support\\Same Tracker Y-036\\tracker-kincal-data-archive-20201203-112300-738\\tracker-kincal-data-archive-20201203-112300-738"
#base = "G:\\My Drive\\Project\\Field Service Support\\Y-036\\Investigation\\PJ run\\tracker_raw_data"
#base = "G:\\My Drive\\Project\\Field Service Support\\Same Tracker Y-036\\tracker-kincal-data (2)\\tracker-kincal-data"

tracker_pose_path = "C:\\Calibration3\\TrackerPostures.csv"
PT_LIST = Yomiread.read_csv(tracker_pose_path, 7, 1000)

# RND ball bar measurements
sb_bb = 186.619/1000
mb_bb = 265.512/1000
lb_bb = 344.575/1000

# 008802
#sb_bb = 202.972 / 1000
#mb_bb = 265.755 / 1000
#lb_bb = 356.994 / 1000


# 008801
#sb_bb = 201.578/1000
#mb_bb = 267.011/1000
#lb_bb = 355.728/1000

#for folder in os.listdir(base):
    #print('folder is', folder)
#for file in os.listdir(base):
sb = Yomiread.read_calibration_measurements(base + '\\sb1.m' )[:, 6:13]
mb = Yomiread.read_calibration_measurements(base + '\\mb1.m' )[:, 6:13]
lb = Yomiread.read_calibration_measurements(base + '\\lb1.m' )[:, 6:13]

fig1 = plt.figure()
#for i in range(7):
#    plt.scatter(range(mb.shape[0]), mb[:,i], label = 'joint '+ np.str(i+1))
#    plt.legend()

#plt.scatter(range(800), sb[:800,1], label = 'joint '+ np.str(1+1))
#plt.scatter(range(800), sb[:800,3], label = 'joint '+ np.str(3+1))
#plt.scatter(range(800), sb[:800,5], label = 'joint '+ np.str(5+1))


#plt.scatter(range(sb.shape[0]), sb[:,1], label = 'joint '+ np.str(1+1))
#plt.scatter(range(sb.shape[0]), sb[:,3], label = 'joint '+ np.str(3+1))
#plt.scatter(range(sb.shape[0]), sb[:,5], label = 'joint '+ np.str(5+1))
#plt.legend()
#plt.show()
#s1 = Yomiread.read_calibration_measurements(base + '\\divot_s1_jointangles.m' )[:, 6:13]
#s2 = Yomiread.read_calibration_measurements(base + '\\divot_s2_jointangles.m' )[:, 6:13]
#s3 = Yomiread.read_calibration_measurements(base + '\\divot_s3_jointangles.m')[:, 6:13]
    #s3 = Yomiread.read_calibration_measurements(base + folder + '\\divot_s3_jointangles.m' )[:, 6:13]

#joint_s1 = Yomiread.read_utiltracker_log(base + '\\S1.log')
#joint_s2 = Yomiread.read_utiltracker_log(base + '\\S2.log')
#joint_s3 = Yomiread.read_utiltracker_log(base + '\\S3.log')

#print('joint_s1 is', joint_s1)
# Add 3 degree off on J3
# sb_offset = sb
# mb_offset = mb
# lb_offset = lb
# offset  = 3.1459*np.pi/180
# for i in range(len(sb)):
#     sb_offset[i,:] = sb[i,:] + [0, 0, offset, 0, 0, 0, 0]
# for j in range(len(mb)):
#     mb_offset[i, :] = mb[i, :] + [0, 0, offset, 0, 0, 0, 0]
# for j in range(len(lb)):
#     lb_offset[i, :] = lb[i, :] + [0, 0, offset, 0, 0, 0, 0]
#kdl_check = opt_debug(sb,sb_bb, mb, mb_bb, lb, lb_bb)
test = opt_bb_debug_single_ball(sb, sb_bb, mb, mb_bb, lb, lb_bb)
#kdl_tem = opt_debug_2bar_test(sb, sb_bb, lb, lb_bb)
#kdl1 = opt_full(sb, sb_bb, mb, mb_bb, lb, lb_bb, s1, s2, s3)

exit()
# solve pivot calibrations
s1_count = joint_s1.shape[0]
s2_count = joint_s2.shape[0]
s3_count = joint_s3.shape[0]

s1t, s1b, cond1 = AOS_pivot_calibration(joint_s1, kdl_tem, s1_count)
s2t, s2b, cond2 = AOS_pivot_calibration(joint_s2, kdl_tem, s2_count)
s3t, s3b, cond3 = AOS_pivot_calibration(joint_s3, kdl_tem, s3_count)
print('s1t is', s1t)
print('s1b is', s1b)

# remove outliers in pivoting
#s1t_2, s1b_2, rms_s1, cond_s1, mean_s1, std_s1, count_s1 = remove_outlier_pivot(joint_s1, kdl_tem, s1t, s1b, s1_count)
#s2t_2, s2b_2, rms_s2, cond_s2, mean_s2, std_s2, count_s2 = remove_outlier_pivot(joint_s2, kdl_tem, s2t, s2b, s2_count)
#s3t_2, s3b_2, rms_s3, cond_s3, mean_s3, std_s3, count_s3 = remove_outlier_pivot(joint_s3, kdl_tem, s3t, s3b, s3_count)

#print('s1 pivot calibration condition number is', cond_s1)
#print('s1 pivot calibration rms is', rms_s1 * 1000, 'mm')
#print('s1 mean is', mean_s1 * 1000, 'mm')
#print('s1 std is', std_s1 * 1000, 'mm')
#print('s1 count is', count_s1)
#print('s2 pivot calibration condition number is', cond_s2)
#print('s2 pivot calibration rms is', rms_s2 * 1000, 'mm')
#print('s2 mean is', mean_s2 * 1000, 'mm')
#print('s2 std is', std_s2 * 1000, 'mm')
#print('s2 count is', count_s2)
#print('s3 pivot calibration condition number is', cond_s3)
#print('s3 pivot calibration rms is', rms_s3 * 1000, 'mm')
#print('s3 mean is', mean_s3 * 1000, 'mm')
#print('s3 std is', std_s3 * 1000, 'mm')
#print('s3 count is', count_s3)

#print('s1 is', s1b_2)
#print('s2 is', s2b_2)
#print('s3 is', s3b_2)
# solve common base transformation
rot, t = solve_commonbase(s1b, s2b, s3b)
print('rot is', rot)
print('t is', t)

#print('old ball position is', ball)
#ball_tem = np.insert(ball, 3, 1.0)
#ball_new = np.matmul(convert_homo_transform(rot, t), ball_tem)
#print('new ball position is', ball_new)
#s1_tem = np.insert(s1b_2, 3, 1.0)
#print('old s1 is', s1b_2)
#print('new s1 is', np.matmul(convert_homo_transform(rot, t), s1_tem))

kdl_final = convert_Yomi_convention(kdl_tem, rot, t)
kdl_result = base + "\\tracker_kdl.csv"
print('kdl_final is', kdl_final)
Yomiwrite.write_csv_array(kdl_result, kdl_final)




exit()

    #result_file = base + folder + "\\Yomisettings.csv"
result_file = base + "\\Yomisettings_debug.csv"
Yomiwrite.write_csv_array(result_file, kdl1, fmt="%.8f")





