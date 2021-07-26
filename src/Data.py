import Readers as Yomiread
import Kinematics as Yomikin
import numpy as np
import timeit
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import math
#import bb_mc_kdl as mcReading

# Delete after verification
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D

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
    bl_tem = bl
    bl_add = bl
    for i in range(m):
        bl_tem = np.vstack((bl_tem, bl_add))
    bl_f = bl_tem[1:]
    return bl_f

# Estimate ball locations for general bar method
def estBall(joint):
    joint_tem = np.copy(joint)
    m, n = np.shape(joint_tem)
    EE_p_est = np.zeros(3)
    for k in range(m):
        EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(kp_guess_common, joint_tem[k, :])
        # EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(tracker_kdl, joint_tem[k, :])
        EE_p_est_tem = np.matmul(EE_T_est_tem[7], [0, 0, 0, 1])
        # EE_p_est_tem = np.matmul(EE_T_est_tem[6], [0, 0, 0, 1])
        EE_p_est = np.vstack((EE_p_est, EE_p_est_tem[0:3]))
    ball = fitSphere(EE_p_est[1:, 0], EE_p_est[1:, 1], EE_p_est[1:, 2])[0:3]
    return ball


def T_nbase(hole1, hole2, hole3):
    # joint1 = np.copy(joint_m)
    # joint2 = np.copy(joint_l)
    # joint3 = np.copy(joint_r)
    # kdl_tem = np.copy(kdl)
    s1 = np.asarray(hole1)
    s2 = np.asarray(hole2)
    s3 = np.asarray(hole3)
    print('s1 is', s1)

    y_axis = (s3 - s2) / np.linalg.norm(s3 - s2)
    z_axis = -np.cross(y_axis, s1 - s3) / np.linalg.norm(np.cross(y_axis, s1 - s3))
    x_axis = np.cross(y_axis, z_axis)

    rotM = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                     [x_axis[1], y_axis[1], z_axis[1]],
                     [x_axis[2], y_axis[2], z_axis[2]]])
    t = s1
    rotMi = np.transpose(rotM)
    ti = -t
    return rotMi, ti


def T_nbase_no(hole1, hole2, hole3):
    # joint1 = np.copy(joint_m)
    # joint2 = np.copy(joint_l)
    # joint3 = np.copy(joint_r)
    # kdl_tem = np.copy(kdl)
    s1 = np.asarray(hole1)
    s2 = np.asarray(hole2)
    s3 = np.asarray(hole3)
    print('s1 is', s1)

    y_axis = (s3 - s2) / np.linalg.norm(s3 - s2)
    z_axis = -np.cross(s3 - s1, y_axis) / np.linalg.norm(np.cross(y_axis, s1 - s3))
    x_axis = np.cross(y_axis, z_axis)

    rotM = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                     [x_axis[1], y_axis[1], z_axis[1]],
                     [x_axis[2], y_axis[2], z_axis[2]]])
    t = s1
    rotMi = np.transpose(rotM)
    ti = -t
    return rotMi, ti


def ee_nbase(joint, kdl, rotMi, ti):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    rot_tem = np.copy(rotMi)
    ti_tem = np.copy(ti)

    m, n = np.shape(joint_tem)
    eepos = np.zeros(3)
    for i in range(m):
        joint_test = np.insert(joint_tem[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_test)
        eepos_tem = np.matmul(pos[7], [0, 0, 0, 1])
        eepos_tem_n = np.matmul(rot_tem, (eepos_tem[0:3] + ti_tem))
        eepos = np.vstack((eepos, eepos_tem_n))

    return eepos[1:, :]

def ee_nbase_full(joint, kdl, rotMi, ti):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    rot_tem = np.copy(rotMi)
    ti_tem = np.copy(ti)

    m, n = np.shape(joint_tem)
    eepos = np.zeros(3)
    for i in range(m):
        #joint_test = np.insert(joint_tem[i, :], 0, 0.)
        pos = Yomikin.FW_Kinematics_Matrices_no_common(kdl_tem, joint_tem[i,:])
        eepos_tem = np.matmul(pos[6], [0, 0, 0, 1])
        eepos_tem_n = np.matmul(rot_tem, (eepos_tem[0:3] + ti_tem))
        eepos = np.vstack((eepos, eepos_tem_n))

    return eepos[1:, :]

def AOS2(joint, kdl):
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
        A[3 * i, :] = np.append(R_tem[0, 0:3], np.array([1, 0, 0]))
        A[3 * i + 1, :] = np.append(R_tem[1, 0:3], np.array([0, 1, 0]))
        A[3 * i + 2, :] = np.append(R_tem[2, 0:3], np.array([0, 0, 1]))
        b[3 * i] = t_tem[0]
        b[3 * i + 1] = t_tem[1]
        b[3 * i + 2] = t_tem[2]
    t = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), b)
    return t[0:3], t[3:6]

def AOS2_no(joint, kdl):
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
        A[3 * i, :] = np.append(R_tem[0, 0:3], np.array([1, 0, 0]))
        A[3 * i + 1, :] = np.append(R_tem[1, 0:3], np.array([0, 1, 0]))
        A[3 * i + 2, :] = np.append(R_tem[2, 0:3], np.array([0, 0, 1]))
        b[3 * i] = t_tem[0]
        b[3 * i + 1] = t_tem[1]
        b[3 * i + 2] = t_tem[2]
    t = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), -b)
    t_c = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A))
    cn = np.linalg.cond(t_c)
    #print('condition number is', cn)
    print('t right is', t)
    ee1,ee2,ee3,count = eeVector_rtip(kdl_tem,joint_tem,t[0:3])
    #eetip = ee1 + t[0:3]
    ee1_mean = np.mean(ee1, axis=0)
    eetip_norm = np.linalg.norm(ee1-ee1_mean,axis=1)
    fig = plt.figure()
    plt.scatter(count, eetip_norm * 1000, alpha=0.5, c='r', label='rtip')
    plt.legend()
    plt.title('rtip position')
    plt.xlabel('Trial Number')
    plt.ylabel('Positions (mm)')

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(ee1[:, 0] * 1000, ee1[:, 1] * 1000, ee1[:, 2] * 1000, zdir='z', s=20, c='r',
               rasterized=True)
    # ax.scatter(ee2[:, 0] * 1000, ee2[:, 1] * 1000, ee2[:, 2] * 1000, zdir='z', s=20, c='g',
    #            rasterized=True)
    plt.title('End effector scatter')
    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')
    #j_2 = remove_outlier(j, out_indx)
    err_data, count_data, out_indx, rms_data = get_err_pivot(joint_tem,kdl_tem,t[0:3])
    #print('outliers indexes are', out_indx)
    #print('err_data is', err_data)
    #plt.show()


    return t[0:3], t[3:6]

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


def looperror_tracker_common_fixed_ball_full(cal, jointangles, bb, ball):
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

    print('rms is', np.sqrt(np.sum(err ** 2, axis=0))/m)
    return err

def opt_common_fixed_ball_full(joint, bb, ball):
    joint_tem = np.copy(joint)
    bb_tem = np.copy(bb)
    kp_guess_common_tem = np.copy(kp_guess_common)
    kp_fit_common_tem = np.copy(kp_fit_common)
    opt = pack(kp_guess_common_tem, kp_fit_common_tem)
    ball_tem = np.copy(ball)
    # ball = np.array([0.04, 0.04, 0.04])
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror_tracker_common_fixed_ball_full, opt,
                                                         (joint_tem, bb_tem, ball_tem), ftol=1e-20,
                                                         full_output=True)
    print('integer flag is', ier)
    print('message is', mesg)
    jac = infodict["fjac"]
    jacobian_condition_number = np.linalg.cond(jac)
    print('condition number is', jacobian_condition_number)
    tem_p = p[:]
    kdl = tem_p[:]
    print('kdl is', kdl)
    ball_est = ball
    kdl_unpack = unpack(kp_guess_common_tem, kdl, kp_fit_common_tem)
    print('kdl_unpack', kdl_unpack)
    print('check rms')
    Vbe_tem, V1e_tem, V2e_tem, count_tem = eeVector(kdl_unpack, joint_tem)
    rms = np.sqrt(np.sum((Vbe_tem - ball_tem)**2)/np.shape(joint_tem))
    print('rms is', rms)
    return kdl_unpack

def opt_pivot_full(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3, joint_s1, joint_s2, joint_s3):
    print('Optimizing with single ball')
    j1 = np.copy(joint_bb1)
    #b1 = dup_bl(estBall(j1), j1)
    bb1 = dup_bl(np.copy(bb_1), j1)
    j2 = np.copy(joint_bb2)
    #b2 = dup_bl(estBall(j2), j2)
    bb2 = dup_bl(np.copy(bb_2), j2)
    j3 = np.copy(joint_bb3)
    #b3 = dup_bl(estBall(j3), j3)
    bb3 = dup_bl(np.copy(bb_3), j3)
    j = np.vstack((j1, j2, j3))
    bb = np.vstack((bb1,bb2,bb3))
    ball = (np.asarray(estBall(j1)) + np.asarray(estBall(j2)) + np.asarray(estBall(j3)))/3

    kdl_tem = opt_common_fixed_ball_full(j, bb, ball)
    err_data, count_data, out_indx, rms_data = get_err(j, bb, ball, kdl_tem)
    j_2 = remove_outlier(j, out_indx)
    bb_2 = remove_outlier(bb, out_indx)
    print('j size is', np.shape(j))
    print('j size after delete is', np.shape(j_2))
    kdl_tem = opt_common_fixed_ball_full(j_2, bb_2, ball)
    #print('kdl is', kdl_tem)

    #print('checking quality, the scatter plot should show a sphere')
    #print('if not the fitting will be failed, check the inputs')
    #fig_1 = plt.figure()
    #plot_ee_scatter(joint_s1, kdl_tem, fig_1)
    #fig_2 = plt.figure()
    #plot_ee_scatter(joint_s2, kdl_tem, fig_2)
    #fig_3 = plt.figure()
    #plot_ee_scatter(joint_s3, kdl_tem, fig_3)
    #plt.show()
    print('solving pivoting holes and the corresponding transformation')
    s1t, s1b = AOS2(joint_s1, kdl_tem)
    s2t, s2b = AOS2(joint_s2, kdl_tem)
    s3t, s3b = AOS2(joint_s3, kdl_tem)

    s1t_2, s1b_2 = remove_outlier_pivot(joint_s1, kdl_tem, s1t, s1b)
    s2t_2, s2b_2 = remove_outlier_pivot(joint_s2, kdl_tem, s2t, s2b)
    s3t_2, s3b_2 = remove_outlier_pivot(joint_s3, kdl_tem, s3t, s3b)

    rot, t = T_nbase(s1b_2, s2b_2, s3b_2)

    rtip = np.vstack((np.vstack((s1t_2, s2t_2)), s3t_2))
    std = np.std(rtip, axis=0)
    mean = np.mean(rtip, axis=0)
    print('std and mean are', std, mean)
    #print('rms is', rms_data)
    # fig_tem = plt.figure()
    # ee1_mean = np.mean(err_data, axis=0)
    # eetip_norm = np.linalg.norm(err_data-ee1_mean,axis=1)
    # plt.scatter(count_data, eetip_norm * 1000, alpha=0.5, c='r', label='rtip')
    #
    # # plt.scatter(count_data, err_data, alpha=0.5, c='r', label='data')
    # # plt.legend()
    # plt.title('data errors')
    # plt.xlabel('Trial Number')
    # plt.ylabel('Errors (mm)')
    # fig_tem2 = plt.figure()
    # ee2_mean = np.mean(err_data2, axis=0)
    # eetip_norm2 = np.linalg.norm(err_data2-ee2_mean,axis=1)
    # plt.scatter(count_data2, eetip_norm2 * 1000, alpha=0.5, c='r', label='rtip')
    # plt.title('data errors')
    # plt.xlabel('Trial Number')
    # plt.ylabel('Errors (mm)')
    #
    # plt.show()

    return kdl_tem, rot, t

def remove_outlier_pivot(joint, kdl, hole, divot):
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    hole_tem = np.copy(hole)
    divot_tem = np.copy(divot)
    err_data, count_data, out_indx, rms_data = get_err_pivot(joint_tem, kdl_tem, -hole_tem, divot_tem)
    print('outliers indexes are', out_indx)
    print('size of err_data is', np.shape(err_data))
    err_data_no = remove_outlier(err_data, out_indx)
    print('size of err_no is', np.shape(err_data_no))
    joint_s1_no = remove_outlier(joint_tem, out_indx)
    s1t_no, s1b_no = AOS2(joint_s1_no, kdl_tem)
    err_data2, count_data2, out_indx2, rms_data2 = get_err_pivot(joint_s1_no, kdl_tem, -s1t_no, s1b_no)
    print('outliers indexes are', out_indx2)
    print('size of err_data is', np.shape(err_data2))
    err_data_no = remove_outlier(err_data2, out_indx2)
    print('size of err_no is', np.shape(err_data_no))
    joint_s2_no = remove_outlier(joint_s1_no, out_indx2)
    s1t_no2, s1b_no2 = AOS2(joint_s2_no, kdl_tem)
    err_data3, count_data3, out_indx3, rms_data3 = get_err_pivot(joint_s2_no, kdl_tem, -s1t_no2, s1b_no2)
    print('outliers indexes are', out_indx3)
    print('size of err_data is', np.shape(err_data3))
    err_data_no2 = remove_outlier(err_data3, out_indx3)
    print('size of err_no is', np.shape(err_data_no2))
    joint_s3_no = remove_outlier(joint_s2_no,out_indx3)
    s1t_no3, s1b_no3 = AOS2(joint_s3_no, kdl_tem)
    return s1t_no3, s1b_no3


def get_err(joint, bb, ball, kdl):
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
        if err[x,] > 0.0005:
            out_indx = np.append(out_indx, np.array([x]))
    eer_rms = np.var(err)
    return err, count, out_indx, eer_rms

def get_err_pivot(joint, kdl, rtip, divot):
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
        #eepos = eepos[0:3] + rtip_tem
        err[x,] = eepos[0:3]
        count[x] = x+1
    #err_mean = np.mean(err, axis=0)
    #err_std = np.std(err)
    #print('err std is', err_std)
    # fig2 = plt.figure()
    # plt.scatter(count, np.linalg.norm(err-divot_tem, axis=1) * 1000, alpha=0.5, c='r', label='m1-s3')
    # plt.legend()
    # plt.title('ee errors')
    # plt.xlabel('Trial Number')
    # plt.ylabel('Errors (mm)')
    #
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='3d')
    # ax.scatter(err[:, 0] * 1000, err[:, 1] * 1000, err[:, 2] * 1000, zdir='z', s=20, c='r',
    #            rasterized=True)
    # ax.scatter(divot_tem[0] * 1000, divot_tem[1] * 1000, divot_tem[2] * 1000, zdir='z', s=20, c='g',
    #            rasterized=True)
    # plt.title('End effector scatter')
    # ax.set_xlabel('X(m)')
    # ax.set_ylabel('Y(m)')
    # ax.set_zlabel('Z(m)')
    # print('divot is', divot_tem)




    #plt.show()



    for x in range(m):
        if np.linalg.norm(err[x,] - divot_tem) > 0.0003:
            out_indx = np.append(out_indx, np.array([x]))
    eer_rms = np.var(err)
    return err, count, out_indx, eer_rms

def remove_outlier(array, out_indx):
    array_tem = np.copy(array)
    indx_tem = np.copy(out_indx)
    for index in sorted(indx_tem, reverse=True):
        array_tem = np.delete(array_tem, int(index), axis=0)
    return array_tem

def homoT(rot,t):
    rot_tem = np.copy(rot)
    t_tem = np.copy(t)
    t1 = np.array([[0.],[0.],[0.]])
    for i in range(3):
        t1[i,0] = np.matmul(rot_tem, t_tem)[i]
    T_tem = np.hstack((rot_tem, t1))
    T = np.vstack((T_tem, np.array([0, 0, 0, 1])))
    return T

def homoT_new(rot,t):
    rot_tem = np.copy(rot)
    t_tem = np.copy(t)
    t1 = np.array([[0.],[0.],[0.]])

    for i in range(3):
        t1[i,0] = np.matmul(rot_tem, -t_tem)[i]
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
    T = homoT(rot_tem, t_tem)
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



path_sb1 = "C:\\Calibration\\sb1.m"
path_sb2 = "C:\\Calibration\\sb2.m"
path_sb3 = "C:\\Calibration\\sb3.m"
path_mb1 = "C:\\Calibration\\mb1.m"
path_mb2 = "C:\\Calibration\\mb2.m"
path_mb3 = "C:\\Calibration\\mb3.m"
path_lb1 = "C:\\Calibration\\lb1.m"
path_lb2 = "C:\\Calibration\\lb2.m"
path_lb3 = "C:\\Calibration\\lb3.m"
path_ball = "C:\\Calibration\\ball_measurements.txt"
path_Yomi = "C:\\Calibration\\complete-set.json"
path_Yomi_Blue = "C:\\Calibration\\YomiSettings_Blue.json"

path_fm1 = "C:\\Calibration\\F_mb1_Matteo.m"
path_fm2 = "C:\\Calibration\\F_mb2_Matteo.m"
path_fm3 = "C:\\Calibration\\F_mb3_Matteo.m"
path_fs1 = "C:\\Calibration\\F_sb1_Matteo.m"
path_fs2 = "C:\\Calibration\\F_sb2_Matteo.m"
path_fs3 = "C:\\Calibration\\F_sb3_Matteo.m"
path_fl1 = "C:\\Calibration\\F_lb1_Matteo.m"
path_fl2 = "C:\\Calibration\\F_lb2_Matteo.m"
path_fl3 = "C:\\Calibration\\F_lb3_Matteo.m"

path_sb_Xiao_1 = "C:\\Calibration3\\sb_Xiao1.m"
path_mb_Xiao_1 = "C:\\Calibration3\\mb_Xiao1.m"
path_lb_Xiao_1 = "C:\\Calibration3\\lb_Xiao1.m"
path_s1_Xiao_1 = "C:\\Calibration3\\s1_Xiao1.m"
path_s2_Xiao_1 = "C:\\Calibration3\\s2_Xiao1.m"
path_s3_Xiao_1 = "C:\\Calibration3\\s3_Xiao1.m"
path_cup_Xiao_1 = "C:\\Calibration3\\cup_Xiao1.m"

joint_sb_Xiao_1 = Yomiread.read_calibration_measurements(path_sb_Xiao_1)[:, 6:13]
joint_mb_Xiao_1 = Yomiread.read_calibration_measurements(path_mb_Xiao_1)[:, 6:13]
joint_lb_Xiao_1 = Yomiread.read_calibration_measurements(path_lb_Xiao_1)[:, 6:13]
joint_s1_Xiao_1 = Yomiread.read_calibration_measurements(path_s1_Xiao_1)[:, 6:13]
joint_s2_Xiao_1 = Yomiread.read_calibration_measurements(path_s2_Xiao_1)[:, 6:13]
joint_s3_Xiao_1 = Yomiread.read_calibration_measurements(path_s3_Xiao_1)[:, 6:13]
joint_cup_Xiao_1 = Yomiread.read_calibration_measurements(path_cup_Xiao_1)[:, 6:13]

#
path_sb_Xiao_2 = "C:\\Calibration3\\sb_Xiao2.m"
path_mb_Xiao_2 = "C:\\Calibration3\\mb_Xiao2.m"
path_lb_Xiao_2 = "C:\\Calibration3\\lb_Xiao2.m"
path_s1_Xiao_2 = "C:\\Calibration3\\s1_Xiao2.m"
path_s2_Xiao_2 = "C:\\Calibration3\\s2_Xiao2.m"
path_s3_Xiao_2 = "C:\\Calibration3\\s3_Xiao2.m"
path_cup_Xiao_2 = "C:\\Calibration3\\cup_Xiao2.m"

joint_sb_Xiao_2 = Yomiread.read_calibration_measurements(path_sb_Xiao_2)[:, 6:13]
joint_mb_Xiao_2 = Yomiread.read_calibration_measurements(path_mb_Xiao_2)[:, 6:13]
joint_lb_Xiao_2 = Yomiread.read_calibration_measurements(path_lb_Xiao_2)[:, 6:13]
joint_s1_Xiao_2 = Yomiread.read_calibration_measurements(path_s1_Xiao_2)[:, 6:13]
joint_s2_Xiao_2 = Yomiread.read_calibration_measurements(path_s2_Xiao_2)[:, 6:13]
joint_s3_Xiao_2 = Yomiread.read_calibration_measurements(path_s3_Xiao_2)[:, 6:13]
joint_cup_Xiao_2 = Yomiread.read_calibration_measurements(path_cup_Xiao_2)[:, 6:13]

path_sb_Xiao_3 = "C:\\Calibration3\\sb_Xiao3.m"
path_mb_Xiao_3 = "C:\\Calibration3\\mb_Xiao3.m"
path_lb_Xiao_3 = "C:\\Calibration3\\lb_Xiao3.m"
path_s1_Xiao_3 = "C:\\Calibration3\\s1_Xiao3.m"
path_s2_Xiao_3 = "C:\\Calibration3\\s2_Xiao3.m"
path_s3_Xiao_3 = "C:\\Calibration3\\s3_Xiao3.m"
path_cup_Xiao_3 = "C:\\Calibration3\\cup_Xiao3.m"

joint_sb_Xiao_3 = Yomiread.read_calibration_measurements(path_sb_Xiao_3)[:, 6:13]
joint_mb_Xiao_3 = Yomiread.read_calibration_measurements(path_mb_Xiao_3)[:, 6:13]
joint_lb_Xiao_3 = Yomiread.read_calibration_measurements(path_lb_Xiao_3)[:, 6:13]
joint_s1_Xiao_3 = Yomiread.read_calibration_measurements(path_s1_Xiao_3)[:, 6:13]
joint_s2_Xiao_3 = Yomiread.read_calibration_measurements(path_s2_Xiao_3)[:, 6:13]
joint_s3_Xiao_3 = Yomiread.read_calibration_measurements(path_s3_Xiao_3)[:, 6:13]
joint_cup_Xiao_3 = Yomiread.read_calibration_measurements(path_cup_Xiao_3)[:, 6:13]

path_sb_Xiao_4 = "C:\\Calibration\\sb_Xiao_4.m"
path_mb_Xiao_4 = "C:\\Calibration\\mb_Xiao_4.m"
path_lb_Xiao_4 = "C:\\Calibration\\lb_Xiao_4.m"
path_s1_Xiao_4 = "C:\\Calibration\\s1_Xiao_4.m"
path_s2_Xiao_4 = "C:\\Calibration\\s2_Xiao_4.m"
path_s3_Xiao_4 = "C:\\Calibration\\s3_Xiao_4.m"

joint_sb_Xiao_4 = Yomiread.read_calibration_measurements(path_sb_Xiao_4)[:, 6:13]
joint_mb_Xiao_4 = Yomiread.read_calibration_measurements(path_mb_Xiao_4)[:, 6:13]
joint_lb_Xiao_4 = Yomiread.read_calibration_measurements(path_lb_Xiao_4)[:, 6:13]
joint_s1_Xiao_4 = Yomiread.read_calibration_measurements(path_s1_Xiao_4)[:, 6:13]
joint_s2_Xiao_4 = Yomiread.read_calibration_measurements(path_s2_Xiao_4)[:, 6:13]
joint_s3_Xiao_4 = Yomiread.read_calibration_measurements(path_s3_Xiao_4)[:, 6:13]

path_sb_D_1 = "C:\\Calibration\\sb_D_1.m"
path_mb_D_1 = "C:\\Calibration\\mb_D_1.m"
path_lb_D_1 = "C:\\Calibration\\lb_D_1.m"
path_s1_D_1 = "C:\\Calibration\\s1_D_1.m"
path_s2_D_1 = "C:\\Calibration\\s2_D_1.m"
path_s3_D_1 = "C:\\Calibration\\s3_D_1.m"

joint_sb_D_1 = Yomiread.read_calibration_measurements(path_sb_D_1)[:, 6:13]
joint_mb_D_1 = Yomiread.read_calibration_measurements(path_mb_D_1)[:, 6:13]
joint_lb_D_1 = Yomiread.read_calibration_measurements(path_lb_D_1)[:, 6:13]
joint_s1_D_1 = Yomiread.read_calibration_measurements(path_s1_D_1)[:, 6:13]
joint_s2_D_1 = Yomiread.read_calibration_measurements(path_s2_D_1)[:, 6:13]
joint_s3_D_1 = Yomiread.read_calibration_measurements(path_s3_D_1)[:, 6:13]



path_sb_PJ_1 = "C:\\Calibration\\sb_PJ_1.m"
path_mb_PJ_1 = "C:\\Calibration\\mb_PJ_1.m"
path_lb_PJ_1 = "C:\\Calibration\\lb_PJ_1.m"
path_s1_PJ_1 = "C:\\Calibration\\s1_PJ_1.m"
path_s2_PJ_1 = "C:\\Calibration\\s2_PJ_1.m"
path_s3_PJ_1 = "C:\\Calibration\\s3_PJ_1.m"

joint_sb_PJ_1 = Yomiread.read_calibration_measurements(path_sb_PJ_1)[:, 6:13]
joint_mb_PJ_1 = Yomiread.read_calibration_measurements(path_mb_PJ_1)[:, 6:13]
joint_lb_PJ_1 = Yomiread.read_calibration_measurements(path_lb_PJ_1)[:, 6:13]
joint_s1_PJ_1 = Yomiread.read_calibration_measurements(path_s1_PJ_1)[:, 6:13]
joint_s2_PJ_1 = Yomiread.read_calibration_measurements(path_s2_PJ_1)[:, 6:13]
joint_s3_PJ_1 = Yomiread.read_calibration_measurements(path_s3_PJ_1)[:, 6:13]

path_sb_Matteo_3 = "C:\\Calibration3\\sb_Matteo3.m"
path_mb_Matteo_3 = "C:\\Calibration3\\mb_Matteo3.m"
path_lb_Matteo_3 = "C:\\Calibration3\\lb_Matteo3.m"
path_s1_Matteo_3 = "C:\\Calibration3\\s1_Matteo3.m"
path_s2_Matteo_3 = "C:\\Calibration3\\s2_Matteo3.m"
path_s3_Matteo_3 = "C:\\Calibration3\\s3_Matteo3.m"
path_cup_Matteo_3 = "C:\\Calibration3\\cup_Matteo3.m"

joint_sb_Matteo_3 = Yomiread.read_calibration_measurements(path_sb_Matteo_3)[:, 6:13]
joint_mb_Matteo_3 = Yomiread.read_calibration_measurements(path_mb_Matteo_3)[:, 6:13]
joint_lb_Matteo_3 = Yomiread.read_calibration_measurements(path_lb_Matteo_3)[:, 6:13]
joint_s1_Matteo_3 = Yomiread.read_calibration_measurements(path_s1_Matteo_3)[:, 6:13]
joint_s2_Matteo_3 = Yomiread.read_calibration_measurements(path_s2_Matteo_3)[:, 6:13]
joint_s3_Matteo_3 = Yomiread.read_calibration_measurements(path_s3_Matteo_3)[:, 6:13]
joint_cup_Matteo_3 = Yomiread.read_calibration_measurements(path_cup_Matteo_3)[:, 6:13]


path_sb_Matteo_2 = "C:\\Calibration3\\sb_Matteo2.m"
path_mb_Matteo_2 = "C:\\Calibration3\\mb_Matteo2.m"
path_lb_Matteo_2 = "C:\\Calibration3\\lb_Matteo2.m"
path_s1_Matteo_2 = "C:\\Calibration3\\s1_Matteo2.m"
path_s2_Matteo_2 = "C:\\Calibration3\\s2_Matteo2.m"
path_s3_Matteo_2 = "C:\\Calibration3\\s3_Matteo2.m"
path_cup_Matteo_2 = "C:\\Calibration3\\cup_Matteo2.m"

joint_sb_Matteo_2 = Yomiread.read_calibration_measurements(path_sb_Matteo_2)[:, 6:13]
joint_mb_Matteo_2 = Yomiread.read_calibration_measurements(path_mb_Matteo_2)[:, 6:13]
joint_lb_Matteo_2 = Yomiread.read_calibration_measurements(path_lb_Matteo_2)[:, 6:13]
joint_s1_Matteo_2 = Yomiread.read_calibration_measurements(path_s1_Matteo_2)[:, 6:13]
joint_s2_Matteo_2 = Yomiread.read_calibration_measurements(path_s2_Matteo_2)[:, 6:13]
joint_s3_Matteo_2 = Yomiread.read_calibration_measurements(path_s3_Matteo_2)[:, 6:13]
joint_cup_Matteo_2 = Yomiread.read_calibration_measurements(path_cup_Matteo_2)[:, 6:13]


path_sb_Matteo_1 = "C:\\Calibration3\\sb_Matteo1.m"
path_mb_Matteo_1 = "C:\\Calibration3\\mb_Matteo1.m"
path_lb_Matteo_1 = "C:\\Calibration3\\lb_Matteo1.m"
path_s1_Matteo_1 = "C:\\Calibration3\\s1_Matteo1.m"
path_s2_Matteo_1 = "C:\\Calibration3\\s2_Matteo1.m"
path_s3_Matteo_1 = "C:\\Calibration3\\s3_Matteo1.m"
path_cup_Matteo_1 = "C:\\Calibration3\\cup_Matteo1.m"

joint_sb_Matteo_1 = Yomiread.read_calibration_measurements(path_sb_Matteo_1)[:, 6:13]
joint_mb_Matteo_1 = Yomiread.read_calibration_measurements(path_mb_Matteo_1)[:, 6:13]
joint_lb_Matteo_1 = Yomiread.read_calibration_measurements(path_lb_Matteo_1)[:, 6:13]
joint_s1_Matteo_1 = Yomiread.read_calibration_measurements(path_s1_Matteo_1)[:, 6:13]
joint_s2_Matteo_1 = Yomiread.read_calibration_measurements(path_s2_Matteo_1)[:, 6:13]
joint_s3_Matteo_1 = Yomiread.read_calibration_measurements(path_s3_Matteo_1)[:, 6:13]
joint_cup_Matteo_1 = Yomiread.read_calibration_measurements(path_cup_Matteo_1)[:, 6:13]

path_poses = "C:\\Calibration3\\TrackerPostures.csv"
joint_poses =  Yomiread.read_csv(path_poses, flag=0)
#print('joint_poses are', joint_poses[1,:])

path_Yomi_Good = "C:\\Calibration4\\YomiSettings_Blue.json"
path_Yomi_Xiao = "C:\\Calibration4\\YomiSettings_Xiao.json"
kdl_Yomi_Good = Yomiread.read_YomiSettings_tracker_kdl(path_Yomi_Good)
kdl_Yomi_Xiao = Yomiread.read_YomiSettings_tracker_kdl(path_Yomi_Xiao)


path_sb_Xiao35_1 = "C:\\Calibration5\\Xiao35_sb1.m"
path_mb_Xiao35_1 = "C:\\Calibration5\\Xiao35_mb1.m"
path_lb_Xiao35_1 = "C:\\Calibration5\\Xiao35_lb1.m"
path_s1_Xiao35_1 = "C:\\Calibration5\\Xiao35_s1_1.m"
path_s2_Xiao35_1 = "C:\\Calibration5\\Xiao35_s2_1.m"
path_s3_Xiao35_1 = "C:\\Calibration5\\Xiao35_s3_1.m"

joint_sb_Xiao35_1 = Yomiread.read_calibration_measurements(path_sb_Xiao35_1)[:, 6:13]
joint_mb_Xiao35_1 = Yomiread.read_calibration_measurements(path_mb_Xiao35_1)[:, 6:13]
joint_lb_Xiao35_1 = Yomiread.read_calibration_measurements(path_lb_Xiao35_1)[:, 6:13]
joint_s1_Xiao35_1 = Yomiread.read_calibration_measurements(path_s1_Xiao35_1)[:, 6:13]
joint_s2_Xiao35_1 = Yomiread.read_calibration_measurements(path_s2_Xiao35_1)[:, 6:13]
joint_s3_Xiao35_1 = Yomiread.read_calibration_measurements(path_s3_Xiao35_1)[:, 6:13]

path_sb_Xiao35_2 = "C:\\Calibration5\\Xiao35_sb2.m"
path_mb_Xiao35_2 = "C:\\Calibration5\\Xiao35_mb2.m"
path_lb_Xiao35_2 = "C:\\Calibration5\\Xiao35_lb2.m"
path_s1_Xiao35_2 = "C:\\Calibration5\\Xiao35_s1_2.m"
path_s2_Xiao35_2 = "C:\\Calibration5\\Xiao35_s2_2.m"
path_s3_Xiao35_2 = "C:\\Calibration5\\Xiao35_s3_2.m"

joint_sb_Xiao35_2 = Yomiread.read_calibration_measurements(path_sb_Xiao35_2)[:, 6:13]
joint_mb_Xiao35_2 = Yomiread.read_calibration_measurements(path_mb_Xiao35_2)[:, 6:13]
joint_lb_Xiao35_2 = Yomiread.read_calibration_measurements(path_lb_Xiao35_2)[:, 6:13]
joint_s1_Xiao35_2 = Yomiread.read_calibration_measurements(path_s1_Xiao35_2)[:, 6:13]
joint_s2_Xiao35_2 = Yomiread.read_calibration_measurements(path_s2_Xiao35_2)[:, 6:13]
joint_s3_Xiao35_2 = Yomiread.read_calibration_measurements(path_s3_Xiao35_2)[:, 6:13]


path_sb_Xiao35_3 = "C:\\Calibration5\\Xiao35_sb3.m"
path_mb_Xiao35_3 = "C:\\Calibration5\\Xiao35_mb3.m"
path_lb_Xiao35_3 = "C:\\Calibration5\\Xiao35_lb3.m"
path_s1_Xiao35_3 = "C:\\Calibration5\\Xiao35_s1_3.m"
path_s2_Xiao35_3 = "C:\\Calibration5\\Xiao35_s2_3.m"
path_s3_Xiao35_3 = "C:\\Calibration5\\Xiao35_s3_3.m"

joint_sb_Xiao35_3 = Yomiread.read_calibration_measurements(path_sb_Xiao35_3)[:, 6:13]
joint_mb_Xiao35_3 = Yomiread.read_calibration_measurements(path_mb_Xiao35_3)[:, 6:13]
joint_lb_Xiao35_3 = Yomiread.read_calibration_measurements(path_lb_Xiao35_3)[:, 6:13]
joint_s1_Xiao35_3 = Yomiread.read_calibration_measurements(path_s1_Xiao35_3)[:, 6:13]
joint_s2_Xiao35_3 = Yomiread.read_calibration_measurements(path_s2_Xiao35_3)[:, 6:13]
joint_s3_Xiao35_3 = Yomiread.read_calibration_measurements(path_s3_Xiao35_3)[:, 6:13]



# Convert the units in ball_measurements to meter
ball_1, ball_2, ball_3, bb_s, bb_m, bb_l = Yomiread.read_ball_measurements(path_ball)
ball_true1 = ball_1 / 1000;
ball_true2 = ball_2 / 1000;
ball_true3 = ball_3 / 1000;
bb_s = bb_s / 1000;
bb_m = bb_m / 1000;
bb_l = bb_l / 1000;
tracker_kdl = Yomiread.read_YomiSettings_tracker_kdl(path_Yomi)
tracker_kdl_blue = Yomiread.read_YomiSettings_tracker_kdl(path_Yomi_Blue)


# Select only the joint angles column 7-13 of each row in '.m' file
joint_sb1 = Yomiread.read_calibration_measurements(path_sb1)[:, 6:13]
joint_mb1 = Yomiread.read_calibration_measurements(path_mb1)[:, 6:13]
joint_lb1 = Yomiread.read_calibration_measurements(path_lb1)[:, 6:13]
joint_sb2 = Yomiread.read_calibration_measurements(path_sb2)[:, 6:13]
joint_mb2 = Yomiread.read_calibration_measurements(path_mb2)[:, 6:13]
joint_lb2 = Yomiread.read_calibration_measurements(path_lb2)[:, 6:13]
joint_sb3 = Yomiread.read_calibration_measurements(path_sb3)[:, 6:13]
joint_mb3 = Yomiread.read_calibration_measurements(path_mb3)[:, 6:13]
joint_lb3 = Yomiread.read_calibration_measurements(path_lb3)[:, 6:13]

joint_fm1 = Yomiread.read_calibration_measurements(path_fm1)[:, 6:13]
joint_fm2 = Yomiread.read_calibration_measurements(path_fm2)[:, 6:13]
joint_fm3 = Yomiread.read_calibration_measurements(path_fm3)[:, 6:13]
joint_fs1 = Yomiread.read_calibration_measurements(path_fs1)[:, 6:13]
joint_fs2 = Yomiread.read_calibration_measurements(path_fs2)[:, 6:13]
joint_fs3 = Yomiread.read_calibration_measurements(path_fs3)[:, 6:13]
joint_fl1 = Yomiread.read_calibration_measurements(path_fl1)[:, 6:13]
joint_fl2 = Yomiread.read_calibration_measurements(path_fl2)[:, 6:13]
joint_fl3 = Yomiread.read_calibration_measurements(path_fl3)[:, 6:13]

ball_fm1 = dup_bl(estBall(joint_fm1), joint_fm1)
ball_fm2 = dup_bl(estBall(joint_fm2), joint_fm2)
ball_fm3 = dup_bl(estBall(joint_fm3), joint_fm3)
ball_fs1 = dup_bl(estBall(joint_fs1), joint_fs1)
ball_fs2 = dup_bl(estBall(joint_fs2), joint_fs2)
ball_fs3 = dup_bl(estBall(joint_fs3), joint_fs3)

# Old bars
#bb_fm = 344.596 / 1000
#bb_fs = 186.629 / 1000
#bb_fl = 490.972 / 1000
# New bars
bb_fm = 265.487/ 1000
bb_fs = 186.629 / 1000
bb_fl = 344.596 / 1000




bb_fm1 = dup_bl(bb_fm, joint_fm1)
bb_fm2 = dup_bl(bb_fm, joint_fm2)
bb_fm3 = dup_bl(bb_fm, joint_fm3)
bb_fs1 = dup_bl(bb_fs, joint_fs1)
bb_fs2 = dup_bl(bb_fs, joint_fs2)
bb_fs3 = dup_bl(bb_fs, joint_fs3)

path_Pre1 = "C:\\Calibration\\Pre_kdl_1.json"
path_Pre2 = "C:\\Calibration\\Pre_kdl_2.json"
path_Pre3 = "C:\\Calibration\\Pre_kdl_3.json"
path_Pre4 = "C:\\Calibration\\Pre_kdl_4.json"
pre_kdl_1 = Yomiread.read_YomiSettings_tracker_kdl(path_Pre1)
pre_kdl_2 = Yomiread.read_YomiSettings_tracker_kdl(path_Pre2)
pre_kdl_3 = Yomiread.read_YomiSettings_tracker_kdl(path_Pre3)
pre_kdl_4 = Yomiread.read_YomiSettings_tracker_kdl(path_Pre4)

path_Art1 = "C:\\Calibration\\Artois_1.json"
path_Art2 = "C:\\Calibration\\Artois_2.json"
path_Art3 = "C:\\Calibration\\Artois_3.json"
path_Art4 = "C:\\Calibration\\Artois_4.json"
path_Art5 = "C:\\Calibration\\Artois_5.json"
art_kdl_1 = Yomiread.read_YomiSettings_tracker_kdl(path_Art1)
art_kdl_2 = Yomiread.read_YomiSettings_tracker_kdl(path_Art2)
art_kdl_3 = Yomiread.read_YomiSettings_tracker_kdl(path_Art3)
art_kdl_4 = Yomiread.read_YomiSettings_tracker_kdl(path_Art4)
art_kdl_5 = Yomiread.read_YomiSettings_tracker_kdl(path_Art5)

path_P2_1 = "C:\\Calibration2\\P2_0L_1.m"
path_P2_2 = "C:\\Calibration2\\P2_0L_2.m"
path_P5_1 = "C:\\Calibration2\\P5_0L_1.m"
path_P5_2 = "C:\\Calibration2\\P5_0L_2.m"
path_P5_3 = "C:\\Calibration2\\P5_0L_3.m"
path_P6_1 = "C:\\Calibration2\\P6_0L_1.m"
path_P6_2 = "C:\\Calibration2\\P6_0L_2.m"
path_P6b_1 = "C:\\Calibration2\\P6b_0L_1.m"
path_P6b_2 = "C:\\Calibration2\\P6b_0L_2.m"

path_P7a_1 = "C:\\Calibration2\\P7a_0L_1.m"
path_P7b_1 = "C:\\Calibration2\\P7b_0L_1.m"
path_P7a_2 = "C:\\Calibration2\\P7a_0L_2.m"
path_P7b_2 = "C:\\Calibration2\\P7b_0L_2.m"
path_P8a_1 = "C:\\Calibration2\\P8a_0L_1.m"
path_P8b_1 = "C:\\Calibration2\\P8b_0L_1.m"
path_P9a_1 = "C:\\Calibration2\\P9a_0L_1.m"
path_P9b_1 = "C:\\Calibration2\\P9b_0L_1.m"

path_pin_m1 = "C:\\Calibration\\Pin_M1.m"
path_pin_m2 = "C:\\Calibration\\Pin_M2.m"
path_pin_m3 = "C:\\Calibration\\Pin_M3.m"
path_pin_m4 = "C:\\Calibration\\Pin_M1_Matteo.m"
path_pin_l1 = "C:\\Calibration\\Pin_L1.m"
path_pin_l2 = "C:\\Calibration\\Pin_L2.m"
path_pin_l3 = "C:\\Calibration\\Pin_L3.m"
path_pin_l4 = "C:\\Calibration\\Pin_L1_Matteo.m"
path_pin_r1 = "C:\\Calibration\\Pin_R1.m"
path_pin_r2 = "C:\\Calibration\\Pin_R2.m"
path_pin_r3 = "C:\\Calibration\\Pin_R3.m"
path_pin_r4 = "C:\\Calibration\\Pin_R1_Matteo.m"

# Select only the joint angles column 7-13 of each row in '.m' file
joint_P2_1 = Yomiread.read_calibration_measurements(path_P2_1)[:, 6:13]
joint_P2_2 = Yomiread.read_calibration_measurements(path_P2_2)[:, 6:13]
#joint_P5_1 = Yomiread.read_calibration_measurements(path_P5_1)[:, 6:13]
#joint_cup_Xiao1 = Yomiread.read_calibration_measurements(path_P5_1)[:, 6:13]
#joint_P5_2 = Yomiread.read_calibration_measurements(path_P5_2)[:, 6:13]
#joint_cup_Xiao2 = Yomiread.read_calibration_measurements(path_P5_2)[:, 6:13]
#joint_P5_3 = Yomiread.read_calibration_measurements(path_P5_3)[:, 6:13]
#joint_cup_Xiao3 = Yomiread.read_calibration_measurements(path_P5_3)[:, 6:13]
#joint_P6_1 = Yomiread.read_calibration_measurements(path_P6_1)[:, 6:13]
#joint_cup_Xiao4 = Yomiread.read_calibration_measurements(path_P6_1)[:, 6:13]
#joint_P6_2 = Yomiread.read_calibration_measurements(path_P6_2)[:, 6:13]
#joint_cup_Matteo1 = Yomiread.read_calibration_measurements(path_P6_2)[:, 6:13]
joint_P6b_1 = Yomiread.read_calibration_measurements(path_P6b_1)[:, 6:13]
joint_P6b_2 = Yomiread.read_calibration_measurements(path_P6b_2)[:, 6:13]

#joint_P7a_1 = Yomiread.read_calibration_measurements(path_P7a_1)[:, 6:13]
#joint_cup_Matteo2 = Yomiread.read_calibration_measurements(path_P7a_1)[:, 6:13]
joint_P7b_1 = Yomiread.read_calibration_measurements(path_P7b_1)[:, 6:13]
#joint_P7a_2 = Yomiread.read_calibration_measurements(path_P7a_2)[:, 6:13]
#joint_cup_Matteo3 = Yomiread.read_calibration_measurements(path_P7a_2)[:, 6:13]
joint_P7b_2 = Yomiread.read_calibration_measurements(path_P7b_2)[:, 6:13]
#joint_P8a_1 = Yomiread.read_calibration_measurements(path_P8a_1)[:, 6:13]
#joint_cup_Daniel1 = Yomiread.read_calibration_measurements(path_P8a_1)[:, 6:13]
joint_P8b_1 = Yomiread.read_calibration_measurements(path_P8b_1)[:, 6:13]
#joint_P9a_1 = Yomiread.read_calibration_measurements(path_P9a_1)[:, 6:13]
#joint_cup_PJ1 = Yomiread.read_calibration_measurements(path_P9a_1)[:, 6:13]
joint_P9b_1 = Yomiread.read_calibration_measurements(path_P9b_1)[:, 6:13]

joint_pin_m1 = Yomiread.read_calibration_measurements(path_pin_m1)[:, 6:13]
joint_pin_m2 = Yomiread.read_calibration_measurements(path_pin_m2)[:, 6:13]
joint_pin_m3 = Yomiread.read_calibration_measurements(path_pin_m3)[:, 6:13]
joint_pin_m4 = Yomiread.read_calibration_measurements(path_pin_m4)[:, 6:13]
joint_pin_l1 = Yomiread.read_calibration_measurements(path_pin_l1)[:, 6:13]
joint_pin_l2 = Yomiread.read_calibration_measurements(path_pin_l2)[:, 6:13]
joint_pin_l3 = Yomiread.read_calibration_measurements(path_pin_l3)[:, 6:13]
joint_pin_l4 = Yomiread.read_calibration_measurements(path_pin_l4)[:, 6:13]
joint_pin_r1 = Yomiread.read_calibration_measurements(path_pin_r1)[:, 6:13]
joint_pin_r2 = Yomiread.read_calibration_measurements(path_pin_r2)[:, 6:13]
joint_pin_r3 = Yomiread.read_calibration_measurements(path_pin_r3)[:, 6:13]
joint_pin_r4 = Yomiread.read_calibration_measurements(path_pin_r4)[:, 6:13]

joint_pin_l = np.vstack((joint_pin_l1, joint_pin_l2, joint_pin_l3, joint_pin_l4))
joint_pin_m = np.vstack((joint_pin_m1, joint_pin_m2, joint_pin_m3, joint_pin_m4))
joint_pin_r = np.vstack((joint_pin_r1, joint_pin_r2, joint_pin_r3, joint_pin_r4))



kdl_fm1_test = np.array([-6.81754816e-02, 1.06725969e-02, 2.74636822e-03, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, -1.52626479e-04, -1.60000000e-01,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.56743482e+00,
                         -4.23946153e-04, 1.86684755e-04, 8.00000000e-02, -2.37689545e-03,
                         1.56601737e+00, 0.00000000e+00, -2.60092876e-01, 1.92781900e-02,
                         0.00000000e+00, -2.00471544e-02, 1.56864571e+00, 0.00000000e+00,
                         -1.85103330e-02, 1.44162435e-03, -5.00000000e-02, -3.11075836e-02,
                         0.00000000e+00, 1.56835902e+00, 1.10506632e-03, 2.33388575e-01,
                         0.00000000e+00, 6.16905930e-02, 0.00000000e+00, -1.56268386e+00,
                         -3.24798606e-04, 6.27244065e-04, -5.00000000e-02, 6.10890919e-04,
                         0.00000000e+00, 1.57187261e+00, 3.08279937e-02, 2.78189991e-02,
                         1.02369675e-01, 0.00000000e+00, 0.00000000e+00, 3.14159265e+00])

kdl_fm2_test = np.array([-3.46524400e-02, 2.23960448e-02, 1.91584897e-02, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 3.50883507e-04, -1.60000000e-01,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.57057192e+00,
                         -1.92274985e-04, 3.24333494e-05, 8.00000000e-02, 1.12267556e-03,
                         1.56924987e+00, 0.00000000e+00, -2.60890576e-01, 2.10399727e-02,
                         0.00000000e+00, -5.95261826e-03, 1.56500868e+00, 0.00000000e+00,
                         -1.79871640e-02, -9.81780220e-04, -5.00000000e-02, -7.04559856e-02,
                         0.00000000e+00, 1.56854202e+00, -1.38863008e-03, 2.38147555e-01,
                         0.00000000e+00, 2.76746546e-01, 0.00000000e+00, -1.56587612e+00,
                         -5.19497033e-05, 3.01865103e-04, -5.00000000e-02, 2.54667276e-03,
                         0.00000000e+00, 1.57020972e+00, -2.96263319e-03, 4.08939505e-02,
                         1.03211185e-01, 0.00000000e+00, 0.00000000e+00, 3.14159265e+00])

kdl_fs3_test = np.array([-5.19032227e-02, 1.40815475e-02, 1.59532540e-02, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 5.50317609e-04, -1.60000000e-01,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.57131845e+00,
                         -1.05485877e-04, -1.39284400e-04, 8.00000000e-02, 2.79229546e-03,
                         1.56910720e+00, 0.00000000e+00, -2.60232230e-01, 2.09650289e-02,
                         0.00000000e+00, -4.89184329e-03, 1.56807562e+00, 0.00000000e+00,
                         -1.81726503e-02, -1.86150575e-03, -5.00000000e-02, -6.97688197e-02,
                         0.00000000e+00, 1.57053087e+00, -1.26348579e-03, 2.37832184e-01,
                         0.00000000e+00, 2.78782157e-01, 0.00000000e+00, -1.56502966e+00,
                         1.98198758e-04, -2.18422897e-04, -5.00000000e-02, 4.53722884e-03,
                         0.00000000e+00, 1.57191311e+00, -3.04043035e-03, 4.10079711e-02,
                         1.02876734e-01, 0.00000000e+00, 0.00000000e+00, 3.14159265e+00])

kdl_Xiao_1 = np.array([-6.69677473e-02,  1.50194311e-02,  7.41619537e-03,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  4.55426395e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57154493e+00,
 -7.15499517e-05, -9.39934913e-05,  8.00000000e-02,  1.13507040e-03,
  1.56899819e+00,  0.00000000e+00, -2.60472481e-01,  2.10828001e-02,
  0.00000000e+00, -6.12814123e-03,  1.56565496e+00,  0.00000000e+00,
 -1.80722794e-02, -1.47298598e-03, -5.00000000e-02, -7.01973182e-02,
  0.00000000e+00,  1.56864628e+00, -1.42385428e-03,  2.38006664e-01,
  0.00000000e+00,  2.78134124e-01,  0.00000000e+00, -1.56559310e+00,
  9.66524373e-05, -1.66955599e-04, -5.00000000e-02,  3.37225933e-03,
  0.00000000e+00,  1.57269311e+00, -2.96393076e-03,  4.09929031e-02,
  1.02722014e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Xiao_1 = np.array([[ 9.97117031e-01, -7.58767392e-02,  5.88575328e-04],
 [ 7.58772125e-02,  9.97116825e-01, -8.28432377e-04],
 [-5.24019614e-04,  8.70703487e-04,  9.99999484e-01]])
t_Xiao_1 = np.array([-0.00380798, -0.00976467, -0.0104612])

kdl_Xiao_2 = np.array([-5.65846201e-02,  1.50987011e-02,  1.19978485e-02,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  3.59381369e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57147695e+00,
 -6.70973621e-05, -7.73570761e-05,  8.00000000e-02,  1.18553637e-03,
  1.56851393e+00,  0.00000000e+00, -2.60546160e-01,  2.09939875e-02,
  0.00000000e+00, -6.05717618e-03,  1.56576675e+00,  0.00000000e+00,
 -1.80030342e-02, -1.58466930e-03, -5.00000000e-02, -6.98365287e-02,
  0.00000000e+00,  1.56845082e+00, -1.34593252e-03,  2.37999050e-01,
  0.00000000e+00,  2.78122753e-01,  0.00000000e+00, -1.56536664e+00,
  1.43940768e-04, -9.64606166e-05, -5.00000000e-02,  3.98661651e-03,
  0.00000000e+00,  1.57194753e+00, -3.02530483e-03,  4.09815103e-02,
  1.02942909e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Xiao_2 = np.array([[ 9.97262341e-01, -7.39435419e-02,  4.19449258e-04],
 [ 7.39436773e-02,  9.97262369e-01, -3.17012900e-04],
 [-3.94859904e-04,  3.47160648e-04,  9.99999862e-01]])
t_Xiao_2 = np.array([-0.01417147, -0.00999638, -0.01507932])

kdl_Xiao_3 = np.array([-6.47483399e-02,  1.92372949e-02,  8.11931126e-03,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  2.62791919e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57147352e+00,
 -1.38102746e-05,  1.76658532e-06,  8.00000000e-02, -9.68896057e-05,
  1.56811845e+00,  0.00000000e+00, -2.60435429e-01,  2.09720547e-02,
  0.00000000e+00, -4.94667132e-03,  1.56632148e+00,  0.00000000e+00,
 -1.81998326e-02, -1.37148682e-03, -5.00000000e-02, -7.06704207e-02,
  0.00000000e+00,  1.56838607e+00, -1.36657438e-03,  2.38123912e-01,
  0.00000000e+00,  2.77922348e-01,  0.00000000e+00, -1.56572968e+00,
  1.39257760e-04, -2.80587266e-05, -5.00000000e-02,  3.84820690e-03,
  0.00000000e+00,  1.57207105e+00, -3.05691319e-03,  4.10253984e-02,
  1.02961348e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Xiao_3 = np.array([[ 9.97244732e-01, -7.41813613e-02, -2.64787752e-04],
 [ 7.41814756e-02,  9.97244656e-01,  4.51543096e-04],
 [ 2.30562090e-04, -4.69941319e-04,  9.99999863e-01]])
t_Xiao_3 = np.array([-0.0060282,  -0.01416442, -0.0111639 ])

kdl_Xiao_4 = np.array([-6.48682482e-02,  1.08208203e-02,  1.11487121e-02,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  3.86502396e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57138435e+00,
 -3.64429808e-05, -6.41881787e-05,  8.00000000e-02,  8.60585814e-04,
  1.56858083e+00,  0.00000000e+00, -2.60566752e-01,  2.10492829e-02,
  0.00000000e+00, -5.58652051e-03,  1.56621449e+00,  0.00000000e+00,
 -1.81512786e-02, -1.36589145e-03, -5.00000000e-02, -7.06575784e-02,
  0.00000000e+00,  1.56826079e+00, -1.37169900e-03,  2.38060867e-01,
  0.00000000e+00,  2.77861513e-01,  0.00000000e+00, -1.56520563e+00,
  1.52790940e-04, -3.15637099e-04, -5.00000000e-02,  3.92058566e-03,
  0.00000000e+00,  1.57302172e+00, -3.04441048e-03,  4.09826569e-02,
  1.02965086e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Xiao_4 = np.array([[ 9.97286577e-01, -7.36163377e-02,  3.43221110e-04],
 [ 7.36164454e-02,  9.97286579e-01, -3.12422134e-04],
 [-3.19290434e-04,  3.36841119e-04,  9.99999892e-01]])
t_Xiao_4 = np.array([-0.0060083,  -0.00571496, -0.01425491])

kdl_Matteo_1 = np.array([-2.77837443e-02,  2.17042155e-02,  1.24180951e-02,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  1.17440527e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57138995e+00,
 -1.22413395e-04, -1.53839953e-04,  8.00000000e-02, -1.27730789e-03,
  1.56867410e+00,  0.00000000e+00, -2.60454995e-01,  2.09433405e-02,
  0.00000000e+00, -6.29056818e-03,  1.56628058e+00,  0.00000000e+00,
 -1.81515189e-02, -1.47142987e-03, -5.00000000e-02, -7.01200455e-02,
  0.00000000e+00,  1.56872635e+00, -1.40209702e-03,  2.38050022e-01,
  0.00000000e+00,  2.78033092e-01,  0.00000000e+00, -1.56560430e+00,
 -1.27414614e-05, -1.35240109e-04, -5.00000000e-02,  2.69951012e-03,
  0.00000000e+00,  1.57254836e+00, -3.03278686e-03,  4.10124586e-02,
  1.02957211e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Matteo_1 = np.array([[ 9.97159378e-01, -7.53204393e-02,  8.15064967e-05],
 [ 7.53204750e-02,  9.97159194e-01, -6.06270565e-04],
 [-3.56103872e-05,  6.10687487e-04,  9.99999813e-01]])
t_Matteo_1 = np.array([-0.04300972, -0.01652223, -0.01565626])

kdl_Matteo_2 = np.array([-5.24162834e-02,  2.12735704e-02,  1.43145477e-02,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  3.64192226e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57146536e+00,
 -5.64195096e-05, -1.37689171e-04,  8.00000000e-02,  1.36172535e-04,
  1.56860161e+00,  0.00000000e+00, -2.60389127e-01,  2.08936421e-02,
  0.00000000e+00, -5.64518156e-03,  1.56601704e+00,  0.00000000e+00,
 -1.80400340e-02, -1.51632350e-03, -5.00000000e-02, -6.96387444e-02,
  0.00000000e+00,  1.56879350e+00, -1.38746496e-03,  2.37998197e-01,
  0.00000000e+00,  2.78127410e-01,  0.00000000e+00, -1.56548960e+00,
  5.74209112e-05, -1.85303597e-04, -5.00000000e-02,  3.23717023e-03,
  0.00000000e+00,  1.57256700e+00, -3.01701239e-03,  4.10271788e-02,
  1.02996004e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Matteo_2 = np.array([[ 9.97198621e-01, -7.47979065e-02,  4.28884925e-04],
 [ 7.47981196e-02,  9.97198570e-01, -5.04265022e-04],
 [-3.89965466e-04,  5.34932170e-04,  9.99999781e-01]])
t_Matteo_2 = np.array([-0.01845051, -0.01610768, -0.01746063])

kdl_Matteo_3 = np.array([-4.62456718e-02,  1.22091189e-02,  4.94788227e-03,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  1.96862099e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57084668e+00,
 -1.14912398e-04, -1.55706719e-04,  8.00000000e-02, -1.03754143e-03,
  1.56866691e+00,  0.00000000e+00, -2.60388391e-01,  2.08031266e-02,
  0.00000000e+00, -5.93176955e-03,  1.56604463e+00,  0.00000000e+00,
 -1.82224282e-02, -1.40541774e-03, -5.00000000e-02, -6.96065966e-02,
  0.00000000e+00,  1.56852280e+00, -1.44348527e-03,  2.38142657e-01,
  0.00000000e+00,  2.77943171e-01,  0.00000000e+00, -1.56540731e+00,
  1.74922564e-04, -3.25692164e-04, -5.00000000e-02,  3.77345159e-03,
  0.00000000e+00,  1.57333740e+00, -3.03163070e-03,  4.10009096e-02,
  1.02994273e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Matteo_3 = np.array([[ 9.97215127e-01, -7.45707252e-02,  1.09452212e-03],
 [ 7.45709595e-02,  9.97215695e-01, -1.74720259e-04],
 [-1.07844562e-03,  2.55853250e-04,  9.99999386e-01]])
t_Matteo_3 = np.array([-0.0246151, -0.007072, -0.00808335])

kdl_Daniel_1 = np.array([-4.53424865e-02,  1.33846069e-02,  5.19760338e-03,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  3.79018180e-04, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57109984e+00,
 -3.88615080e-05, -3.25789714e-05,  8.00000000e-02,  2.39865569e-04,
  1.56860559e+00,  0.00000000e+00, -2.60411819e-01,  2.11018819e-02,
  0.00000000e+00, -5.37031644e-03,  1.56615831e+00,  0.00000000e+00,
 -1.82502195e-02, -1.37242053e-03, -5.00000000e-02, -7.06036736e-02,
  0.00000000e+00,  1.56862413e+00, -1.46311278e-03,  2.38096073e-01,
  0.00000000e+00,  2.77874523e-01,  0.00000000e+00, -1.56506007e+00,
  8.02774425e-05, -4.46261111e-04, -5.00000000e-02,  3.05221621e-03,
  0.00000000e+00,  1.57337706e+00, -3.01098816e-03,  4.09644333e-02,
  1.02992029e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_Daniel_1 = np.array([[ 9.97255176e-01, -7.40411021e-02, -1.71593545e-04],
 [ 7.40409308e-02,  9.97254840e-01, -8.50790720e-04],
 [ 2.34115976e-04,  8.35750504e-04,  9.99999623e-01]])
t_Daniel_1 = np.array([-0.02535746, -0.0081894,  -0.00811775])

kdl_PJ_1 = np.array([-1.32114140e-02,  2.20498421e-02,  1.56874555e-02,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00, -6.53532380e-05, -1.60000000e-01,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.57243767e+00,
 -5.87591558e-06, -1.11699669e-04,  8.00000000e-02, -2.81880133e-03,
  1.56927451e+00,  0.00000000e+00, -2.60541954e-01,  2.11695456e-02,
  0.00000000e+00, -5.79178197e-03,  1.56607887e+00,  0.00000000e+00,
 -1.82582027e-02, -1.40600550e-03, -5.00000000e-02, -7.07141006e-02,
  0.00000000e+00,  1.56848400e+00, -1.44148337e-03,  2.38020715e-01,
  0.00000000e+00,  2.78062358e-01,  0.00000000e+00, -1.56544268e+00,
  2.50305334e-04, -2.43647251e-04, -5.00000000e-02,  4.20966901e-03,
  0.00000000e+00,  1.57295212e+00, -2.99608165e-03,  4.10730376e-02,
  1.02699863e-01,  0.00000000e+00,  0.00000000e+00,  3.14159265e+00])
rot_PJ_1 = np.array([[ 0.99711858, -0.07581835,  0.00247376],
 [ 0.07582287,  0.9971197,  -0.00178989],
 [-0.00233093,  0.0019723,   0.99999534]])
t_PJ_1 = np.array([-0.05772301, -0.01678031, -0.0188976])


kdl_no_common = tracker_kdl[6:]  # get rid of the transformation to common base
kdl_common = tracker_kdl  # kdl with the common reference transformation
print('kdl_no_common is', np.shape(kdl_no_common))
joint_verification2 = np.zeros((20, 8))

path_sb_Marco_1 = "C:\\Calibration3\\Marco_nb_sb1.m"
path_mb_Marco_1 = "C:\\Calibration3\\Marco_nb_mb1.m"
path_lb_Marco_1 = "C:\\Calibration3\\Marco_nb_lb1.m"
path_s1_Marco_1 = "C:\\Calibration3\\Marco_s1.m"
path_s2_Marco_1 = "C:\\Calibration3\\Marco_s2.m"
path_s3_Marco_1 = "C:\\Calibration3\\Marco_s3.m"


joint_sb_Marco_1 = Yomiread.read_calibration_measurements(path_sb_Marco_1)[:, 6:13]
joint_mb_Marco_1 = Yomiread.read_calibration_measurements(path_mb_Marco_1)[:, 6:13]
joint_lb_Marco_1 = Yomiread.read_calibration_measurements(path_lb_Marco_1)[:, 6:13]
joint_s1_Marco_1 = Yomiread.read_calibration_measurements(path_s1_Marco_1)[:, 6:13]
joint_s2_Marco_1 = Yomiread.read_calibration_measurements(path_s2_Marco_1)[:, 6:13]
joint_s3_Marco_1 = Yomiread.read_calibration_measurements(path_s3_Marco_1)[:, 6:13]


path_sb_B_1 = "C:\\Calibration5\\3_16_trial1_sb.m"
path_mb_B_1 = "C:\\Calibration5\\3_16_trial1_mb.m"
path_lb_B_1 = "C:\\Calibration5\\3_16_trial1_lb.m"
path_s1_B_1 = "C:\\Calibration5\\3_16_trial1_s1.m"
path_s2_B_1 = "C:\\Calibration5\\3_16_trial1_s2.m"
path_s3_B_1 = "C:\\Calibration5\\3_16_trial1_s3.m"


joint_sb_B_1 = Yomiread.read_calibration_measurements(path_sb_B_1)[:, 6:13]
joint_mb_B_1 = Yomiread.read_calibration_measurements(path_mb_B_1)[:, 6:13]
joint_lb_B_1 = Yomiread.read_calibration_measurements(path_lb_B_1)[:, 6:13]
joint_s1_B_1 = Yomiread.read_calibration_measurements(path_s1_B_1)[:, 6:13]
joint_s2_B_1 = Yomiread.read_calibration_measurements(path_s2_B_1)[:, 6:13]
joint_s3_B_1 = Yomiread.read_calibration_measurements(path_s3_B_1)[:, 6:13]

path_sb_B_2 = "C:\\Calibration5\\3_16_trial2_sb.m"
path_mb_B_2 = "C:\\Calibration5\\3_16_trial2_mb.m"
path_lb_B_2 = "C:\\Calibration5\\3_16_trial2_lb.m"
path_s1_B_2 = "C:\\Calibration5\\3_16_trial2_s1.m"
path_s2_B_2 = "C:\\Calibration5\\3_16_trial2_s2.m"
path_s3_B_2 = "C:\\Calibration5\\3_16_trial2_s3.m"


joint_sb_B_2 = Yomiread.read_calibration_measurements(path_sb_B_2)[:, 6:13]
joint_mb_B_2 = Yomiread.read_calibration_measurements(path_mb_B_2)[:, 6:13]
joint_lb_B_2 = Yomiread.read_calibration_measurements(path_lb_B_2)[:, 6:13]
joint_s1_B_2 = Yomiread.read_calibration_measurements(path_s1_B_2)[:, 6:13]
joint_s2_B_2 = Yomiread.read_calibration_measurements(path_s2_B_2)[:, 6:13]
joint_s3_B_2 = Yomiread.read_calibration_measurements(path_s3_B_2)[:, 6:13]


path_sb_B_3 = "C:\\Calibration5\\3_16_trial3_sb.m"
path_mb_B_3 = "C:\\Calibration5\\3_16_trial3_mb.m"
path_lb_B_3 = "C:\\Calibration5\\3_16_trial3_lb.m"
path_s1_B_3 = "C:\\Calibration5\\3_16_trial3_s1.m"
path_s2_B_3 = "C:\\Calibration5\\3_16_trial3_s2.m"
path_s3_B_3 = "C:\\Calibration5\\3_16_trial3_s3.m"


joint_sb_B_3 = Yomiread.read_calibration_measurements(path_sb_B_3)[:, 6:13]
joint_mb_B_3 = Yomiread.read_calibration_measurements(path_mb_B_3)[:, 6:13]
joint_lb_B_3 = Yomiread.read_calibration_measurements(path_lb_B_3)[:, 6:13]
joint_s1_B_3 = Yomiread.read_calibration_measurements(path_s1_B_3)[:, 6:13]
joint_s2_B_3 = Yomiread.read_calibration_measurements(path_s2_B_3)[:, 6:13]
joint_s3_B_3 = Yomiread.read_calibration_measurements(path_s3_B_3)[:, 6:13]

path_r1_Marco_1 = "C:\\Calibration5\\Marco_rtip1.m"
path_r2_Marco_1 = "C:\\Calibration5\\Marco_rtip2.m"


joint_r1_Marco_1 = Yomiread.read_calibration_measurements(path_r1_Marco_1)[:, 6:13]
joint_r2_Marco_1 = Yomiread.read_calibration_measurements(path_r2_Marco_1)[:, 6:13]

path_sb_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\sb1.m"
path_mb_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\mb1.m"
path_lb_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\lb1.m"
path_s1_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\s1.m"
path_s2_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\s2.m"
path_s3_P_1 = "C:\\Calibration5\\Tracker_neo test\\pj\\s3.m"


joint_sb_P_1 = Yomiread.read_calibration_measurements(path_sb_P_1)[:, 6:13]
joint_mb_P_1 = Yomiread.read_calibration_measurements(path_mb_P_1)[:, 6:13]
joint_lb_P_1 = Yomiread.read_calibration_measurements(path_lb_P_1)[:, 6:13]
joint_s1_P_1 = Yomiread.read_calibration_measurements(path_s1_P_1)[:, 6:13]
joint_s2_P_1 = Yomiread.read_calibration_measurements(path_s2_P_1)[:, 6:13]
joint_s3_P_1 = Yomiread.read_calibration_measurements(path_s3_P_1)[:, 6:13]


#path_r1_Brian_3 = "C:\\Calibration5\\Brian_rtip1_3.m"
#path_r2_Brian_3 = "C:\\Calibration5\\Brian_rtip2_3.m"


#joint_r1_Brian_3 = Yomiread.read_calibration_measurements(path_r1_Brian_3)[:, 6:13]
#joint_r2_Brian_3 = Yomiread.read_calibration_measurements(path_r2_Brian_3)[:, 6:13]


kdl13 = np.array([-7.085177973302199395e-02,
5.369727984357304355e-03,
-3.072379219890799228e-03,
7.702655908961396347e-02,
5.448955069781980225e-04,
8.850355688452237997e-04,
6.006558188425275169e-05,
-1.600000000000000033e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
-1.571394944266405469e+00,
-6.853405586725551521e-05,
5.383262361932578119e-06,
8.000000000000000167e-02,
-1.056028990446535856e-03,
1.569018069095047707e+00,
0.000000000000000000e+00,
-2.602890325610245359e-01,
2.071761512231120109e-02,
0.000000000000000000e+00,
-6.450695708003713162e-03,
1.565657643707985214e+00,
0.000000000000000000e+00,
-1.829574530215484149e-02,
-1.702540670107018919e-03,
-5.000000000000000278e-02,
-7.016780712250282603e-02,
0.000000000000000000e+00,
1.569043178999762977e+00,
-1.373257221742898255e-03,
2.381623839087287087e-01,
0.000000000000000000e+00,
2.782487334495561404e-01,
0.000000000000000000e+00,
-1.565112593015141051e+00,
1.856484211737147551e-04,
-4.557052068913530502e-04,
-5.000000000000000278e-02,
3.611789610651931597e-03,
0.000000000000000000e+00,
1.573651544598057450e+00,
-3.159136494531184988e-03,
4.090905840827446793e-02,
1.027376920907092095e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
3.141592653589793116e+00])

kdl16 = np.array([-7.085189171426843624e-02,
5.369772124158151728e-03,
2.692743885160744227e-02,
7.702700946491071454e-02,
5.452455750339263744e-04,
8.851236035105900565e-04,
6.014792628675121869e-05,
-1.300000000000000044e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
-1.571394539583273797e+00,
-8.656194339929423128e-05,
5.180785162128345587e-06,
8.000000000000000167e-02,
-1.056465799291725176e-03,
1.569017925702148153e+00,
0.000000000000000000e+00,
-2.602889912183562471e-01,
2.071743982547809976e-02,
0.000000000000000000e+00,
-6.451389880798890503e-03,
1.565657628902552112e+00,
0.000000000000000000e+00,
-1.829581626955000678e-02,
-1.702638178688051576e-03,
-5.000000000000000278e-02,
-7.016747750884375079e-02,
0.000000000000000000e+00,
1.569043328410993920e+00,
-1.373264550177542538e-03,
2.381624016947683498e-01,
0.000000000000000000e+00,
2.782488883273289693e-01,
0.000000000000000000e+00,
-1.565112582586579482e+00,
1.856450203635669411e-04,
-4.557081872004236007e-04,
-5.000000000000000278e-02,
3.611723651963225871e-03,
0.000000000000000000e+00,
1.573651537494416219e+00,
-3.159146078654888421e-03,
4.090906737926562620e-02,
1.027376876700244468e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
3.141592653589793116e+00])


path_sb_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\sb_neo2.m"
path_mb_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\mb_neo2.m"
path_lb_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\lb_neo2.m"
path_s1_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\s1_neo2.m"
path_s2_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\s2_neo2.m"
path_s3_neo1 = "C:\\Calibration5\\Tracker_neo test\\trial1\\s3_neo2.m"


joint_sb_neo1 = Yomiread.read_calibration_measurements(path_sb_neo1)[:, 6:13]
joint_mb_neo1 = Yomiread.read_calibration_measurements(path_mb_neo1)[:, 6:13]
joint_lb_neo1 = Yomiread.read_calibration_measurements(path_lb_neo1)[:, 6:13]
joint_s1_neo1 = Yomiread.read_calibration_measurements(path_s1_neo1)[:, 6:13]
joint_s2_neo1 = Yomiread.read_calibration_measurements(path_s2_neo1)[:, 6:13]
joint_s3_neo1 = Yomiread.read_calibration_measurements(path_s3_neo1)[:, 6:13]


path_sb_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\sb_neo1.m"
path_mb_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\mb_neo1.m"
path_lb_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\lb_neo1.m"
path_s1_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\s1_neo1.m"
path_s2_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\s2_neo1.m"
path_s3_neo2 = "C:\\Calibration5\\Tracker_neo test\\trial2\\s3_neo1.m"


joint_sb_neo2 = Yomiread.read_calibration_measurements(path_sb_neo2)[:, 6:13]
joint_mb_neo2 = Yomiread.read_calibration_measurements(path_mb_neo2)[:, 6:13]
joint_lb_neo2 = Yomiread.read_calibration_measurements(path_lb_neo2)[:, 6:13]
joint_s1_neo2 = Yomiread.read_calibration_measurements(path_s1_neo2)[:, 6:13]
joint_s2_neo2 = Yomiread.read_calibration_measurements(path_s2_neo2)[:, 6:13]
joint_s3_neo2 = Yomiread.read_calibration_measurements(path_s3_neo2)[:, 6:13]


path_sb_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\sb_neo3.m"
path_mb_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\mb_neo3.m"
path_lb_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\lb_neo3.m"
path_s1_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\s1_neo3.m"
path_s2_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\s2_neo3.m"
path_s3_neo3 = "C:\\Calibration5\\Tracker_neo test\\trial3\\s3_neo3.m"


joint_sb_neo3 = Yomiread.read_calibration_measurements(path_sb_neo3)[:, 6:13]
joint_mb_neo3 = Yomiread.read_calibration_measurements(path_mb_neo3)[:, 6:13]
joint_lb_neo3 = Yomiread.read_calibration_measurements(path_lb_neo3)[:, 6:13]
joint_s1_neo3 = Yomiread.read_calibration_measurements(path_s1_neo3)[:, 6:13]
joint_s2_neo3 = Yomiread.read_calibration_measurements(path_s2_neo3)[:, 6:13]
joint_s3_neo3 = Yomiread.read_calibration_measurements(path_s3_neo3)[:, 6:13]

path_sb_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\sb1.m"
path_mb_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\mb1.m"
path_lb_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\lb1.m"
path_s1_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\S1.m"
path_s2_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\S2.m"
path_s3_neo_dt = "C:\\Users\\xjboston\\Desktop\Projects\\Calibration_1B\\Tracker_neo test\\DT Calibration\\DT2\\S3.m"


joint_sb_neo_dt = Yomiread.read_calibration_measurements(path_sb_neo_dt)[:, 6:13]
joint_mb_neo_dt = Yomiread.read_calibration_measurements(path_mb_neo_dt)[:, 6:13]
joint_lb_neo_dt = Yomiread.read_calibration_measurements(path_lb_neo_dt)[:, 6:13]
joint_s1_neo_dt = Yomiread.read_calibration_measurements(path_s1_neo_dt)[:, 6:13]
joint_s2_neo_dt = Yomiread.read_calibration_measurements(path_s2_neo_dt)[:, 6:13]
joint_s3_neo_dt = Yomiread.read_calibration_measurements(path_s3_neo_dt)[:, 6:13]


# Verification for tracker_neo
kdl_neo1 = np.array([-0.07076266215448787,
      0.003262101570829262,
      -8.068235592714618e-05,
      0.05555983148539704,
      0.0007769767332564182,
      0.001040820256528661,
      -0.0001104421393784641,
      -0.157,
      0,
      0,
      0,
      -1.567887389133195,
      -0.0002508297186504913,
      -0.0001601761616337641,
      0.08,
      0.009061052729006411,
      1.55875394893358,
      0,
      -0.1994255958249683,
      0.003831866989810776,
      0,
      -0.07269955464279994,
      1.586061073824459,
      0,
      -0.01820388351404811,
      -0.002447479917129419,
      -0.05,
      -0.05404578668214282,
      0,
      1.573011525695218,
      2.733459644591506e-05,
      0.1598494593073328,
      0,
      0.0428712681923088,
      0,
      -1.571532137142845,
      0.0001417526233537728,
      0.0005102683339404128,
      -0.05,
      0.002633396212597693,
      0,
      1.571922152307975,
      0.03436886517504933,
      0.02271174432414215,
      0.102740805073044,
      0,
      0,
      3.141592653589793])
kdl_neo2 = np.array([      -0.07077570008878786,
      0.003244015919698589,
      6.871862840722367e-05,
      0.05468625959055579,
      0.0002342451247221019,
      0.0005907246255908258,
      7.974858776202499e-05,
      -0.157,
      0,
      0,
      0,
      -1.567157750096354,
      -0.0003356039355863751,
      -6.644752136494554e-05,
      0.08,
      0.01117040293555823,
      1.558238818877054,
      0,
      -0.1994147744354446,
      0.003668276082473878,
      0,
      -0.07128707394059711,
      1.586023002837966,
      0,
      -0.01823517882100695,
      -0.001950696432644225,
      -0.05,
      -0.05372987536717638,
      0,
      1.572894737851265,
      -7.410847254440785e-05,
      0.1598166324734785,
      0,
      0.04163158471171131,
      0,
      -1.571566982551604,
      8.148175913106986e-05,
      0.0008585042316255103,
      -0.05,
      0.001736283531292936,
      0,
      1.569990320839949,
      0.03445636138231586,
      0.02279023812650382,
      0.1027681714776704,
      0,
      0,
      3.141592653589793
])
kdl_neo3 = np.array([-0.07084489885892351,
      0.003391542184218747,
      -0.0001360507649887866,
      0.05696093287740088,
      0.002076469316195883,
      0.0003078954596931294,
      -7.500380085927218e-05,
      -0.157,
      0,
      0,
      0,
      -1.567348084735402,
      -0.0002563090174807953,
      -0.0001705617798027472,
      0.08,
      0.009718223157097255,
      1.559437663890741,
      0,
      -0.1994386140427413,
      0.003792119105100635,
      0,
      -0.07234111566431491,
      1.585748534431453,
      0,
      -0.01823763282882155,
      -0.002094314770418985,
      -0.05,
      -0.05398099248049155,
      0,
      1.573034962043798,
      1.643504992665131e-05,
      0.1597643165845136,
      0,
      0.0421385979486321,
      0,
      -1.570857193109786,
      0.0003515543437483631,
      0.0001662755720680635,
      -0.05,
      0.003719194050428312,
      0,
      1.573246010200209,
      0.03445011184271694,
      0.02280290794594702,
      0.1027091110113724,
      0,
      0,
      3.141592653589793
])
kdl_neo0 = np.array([
    -0.07084687650985703,
    -0.0007372637395036036,
    0,
    0,
    0.000609091716978852,
    0.0004403738015915028,
    7.105323858167193e-05,
    -0.1570532651329002,
    0,
    0.05678893206054594,
    0,
    -1.567448157625704,
    -0.0001965034834216875,
    -0.000124772918686974,
    0.08,
    0.01071446945400324,
    1.559478489964231,
    0,
    -0.1993656092674445,
    0.003713357299066687,
    0,
    -0.07158574547742802,
    1.586295130499281,
    0,
    -0.01824265045641332,
    -0.002043911030908435,
    -0.05,
    -0.05365749700457934,
    0,
    1.573008754243513,
    6.369560854117479e-06,
    0.1599444950916375,
    0,
    0.04217350289708872,
    0,
    -1.571088113057816,
    2.870523793834295e-05,
    0.0004051599329929604,
    -0.05,
    0.001678932737376521,
    0,
    1.572011809859696,
    0.03439573810108439,
    0.02271312663056718,
    0.1027752905768159,
    0,
    0,
    3.14159265358979
])
kdl_pj_neo = np.array([
-7.085685566406195457e-02,
3.261025997676364561e-03,
1.003176857767633064e-04,
5.543108990266429337e-02,
1.892587718534845167e-03,
4.444681539180618903e-04,
7.731949380743100410e-05,
-1.570000000000000007e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
-1.567768764732173414e+00,
-1.451529302571501288e-04,
-3.195649964565266179e-05,
8.000000000000000167e-02,
9.753821455526244522e-03,
1.558915128155225593e+00,
0.000000000000000000e+00,
-1.992045491273935487e-01,
3.652314275334947805e-03,
0.000000000000000000e+00,
-7.181119240205886822e-02,
1.586040742076487264e+00,
0.000000000000000000e+00,
-1.847696290829280857e-02,
-2.462705286999473595e-03,
-5.000000000000000278e-02,
-5.463200128820005358e-02,
0.000000000000000000e+00,
1.572895873659383437e+00,
6.280250850096388328e-05,
1.601648953905278894e-01,
0.000000000000000000e+00,
4.285127269291633123e-02,
0.000000000000000000e+00,
-1.571361433218740888e+00,
1.058963682842213334e-04,
4.478618407352619808e-04,
-5.000000000000000278e-02,
2.579024893478978377e-03,
0.000000000000000000e+00,
1.572149963938037409e+00,
3.437512520850977832e-02,
2.271684132194912165e-02,
1.027804560246273202e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
3.141592653589793116e+00
])
kdl_dt_neo = np.array([
-7.072926626866407607e-02,
3.092285409802574426e-03,
2.219996049167483249e-04,
5.436137318370306420e-02,
1.767872717810417202e-03,
8.785553856958791454e-04,
6.917175638928669093e-07,
-1.570000000000000007e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
-1.566875770301317417e+00,
-2.015856375536725909e-04,
3.896486974112802360e-05,
8.000000000000000167e-02,
1.100104878286202459e-02,
1.558638652193581553e+00,
0.000000000000000000e+00,
-1.992173834862282777e-01,
3.524859828442293910e-03,
0.000000000000000000e+00,
-7.054566564942657547e-02,
1.585301666078739569e+00,
0.000000000000000000e+00,
-1.862055383613077375e-02,
-1.891871957206063158e-03,
-5.000000000000000278e-02,
-5.453189587956955298e-02,
0.000000000000000000e+00,
1.572739442783278818e+00,
1.689419835480945658e-06,
1.601102528039440309e-01,
0.000000000000000000e+00,
4.080822677736514054e-02,
0.000000000000000000e+00,
-1.571583503271991811e+00,
1.938672349279591044e-04,
5.053039648533552220e-04,
-5.000000000000000278e-02,
2.605938877310742634e-03,
0.000000000000000000e+00,
1.572220886497988968e+00,
3.435554147446321510e-02,
2.277188192660597327e-02,
1.027165493410518854e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
3.141592653589793116e+00
])
kdl_dt_neo2 = np.array([
-7.079626792975324268e-02,
3.148748415010487606e-03,
2.988672689634303274e-04,
5.589103587822758651e-02,
-6.525978012471807659e-06,
1.652015505068687615e-04,
-5.827162768474178025e-05,
-1.570000000000000007e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
-1.567290304079518748e+00,
-1.692073957392658146e-04,
3.367676309489546732e-05,
8.000000000000000167e-02,
1.035408049919441832e-02,
1.558708415106136336e+00,
0.000000000000000000e+00,
-1.994201096864607614e-01,
3.723498187260752476e-03,
0.000000000000000000e+00,
-6.986119201227457276e-02,
1.586018352851508384e+00,
0.000000000000000000e+00,
-1.836062806458730115e-02,
-1.783554262663788418e-03,
-5.000000000000000278e-02,
-5.435139950068711145e-02,
0.000000000000000000e+00,
1.572819503655841444e+00,
-8.347982327035635843e-06,
1.599825227555601859e-01,
0.000000000000000000e+00,
4.111672872978104631e-02,
0.000000000000000000e+00,
-1.571040087623325254e+00,
2.768219257531951967e-04,
3.219923140618693959e-04,
-5.000000000000000278e-02,
3.111593446887848946e-03,
0.000000000000000000e+00,
1.572592343542655868e+00,
3.432816771450639126e-02,
2.282178337064117227e-02,
1.027908443625001750e-01,
0.000000000000000000e+00,
0.000000000000000000e+00,
3.141592653589793116e+00
])
# Check rubytip optimization

# Kinova Yomi
