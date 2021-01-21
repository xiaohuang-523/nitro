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
import fileDialog

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


# Estimate common base sphere locations for general bar method
def est_sphere_CB_debug(joint, barlength):
    joint_tem = joint
    m, n = np.shape(joint_tem)
    EE_p_est_x = []
    EE_p_est_y = []
    EE_p_est_z = []
    EE_p_est = []
    for k in range(m):
        EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(kp_guess_common, joint_tem[k, :])
        # EE_T_est_tem = Yomikin.FW_Kinematics_Matrices(tracker_kdl, joint_tem[k, :])
        EE_p_est_tem = np.matmul(EE_T_est_tem[7], [0, 0, 0, 1])
        # EE_p_est_tem = np.matmul(EE_T_est_tem[6], [0, 0, 0, 1])
        EE_p_est_x.append(EE_p_est_tem[0])
        EE_p_est_y.append(EE_p_est_tem[1])
        EE_p_est_z.append(EE_p_est_tem[2])
        EE_p_est.append(EE_p_est_tem[0:3])
    ball = fitSphere(np.array(EE_p_est_x), np.array(EE_p_est_y), np.array(EE_p_est_z))[0:3]
    diff_vec = np.asarray(EE_p_est) - np.asarray(ball)
    # convert distance units from m to mm
    diff = np.sqrt(np.sum(diff_vec**2, axis=1)) * 1000

    # To check, we want the point cloud to be similar to a shpere. Use standard deviation of estimated radius.
    mean = np.mean(diff)
    variance = np.var(diff)
    stdev = np.std(diff)
    print('variance is', variance)
    print('stdev is', stdev)
    print('mean is', mean)

    # To check the bar-ball length measurement, compare mean with the known bar length.
    bar_length_diff = mean - barlength * 1000
    print('bar_length_diff is', bar_length_diff)

    return np.asarray(ball), mean, variance, stdev, barlength*1000, bar_length_diff


def opt_full_bar_ball_data_check(joint_bb1, bb_1, joint_bb2, bb_2, joint_bb3, bb_3):
    print('Checking ball bar data quality')

    bar1_param = est_sphere_CB_debug(joint_bb1, bb_1)

    bar2_param = est_sphere_CB_debug(joint_bb2, bb_2)

    bar3_param = est_sphere_CB_debug(joint_bb3, bb_3)

    result = []
    result.append(bar1_param[1:])
    result.append(bar2_param[1:])
    result.append(bar3_param[1:])

    return np.asarray(result)



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
#base = "G:\\My Drive\\Project\\Field Service Support\\tracker-kincal-data - recalibration for Xiao\\tracker-kincal-data - recalibration for Xiao\\"
#base = "G:\\My Drive\\Project\\single_ball PT Calibration ball bar data check\\Error Tracker Data\\tracker-kincal-data\\tracker-kincal-data"
base = fileDialog.select_folder()
# RND ball bar measurements
#sb_bb = 186.619/1000
#mb_bb = 265.512/1000
#lb_bb = 344.575/1000

barlength = Yomiread.read_barset(base + '\\barset_measurements.txt' )
sb_bb = barlength[0]/1000
mb_bb = barlength[1]/1000
lb_bb = barlength[2]/1000
print('barlength is', np.shape(barlength))
#
# sb_bb = 201.813/1000
# mb_bb = 265.456/1000
# lb_bb = 353.918/1000


sb = Yomiread.read_calibration_measurements(base + '\\sb1.m' )[:, 6:13]
mb = Yomiread.read_calibration_measurements(base + '\\mb1.m' )[:, 6:13]
lb = Yomiread.read_calibration_measurements(base + '\\lb1.m' )[:, 6:13]
#s1 = Yomiread.read_calibration_measurements(base + '\\divot_s1_jointangles.m' )[:, 6:13]
#s2 = Yomiread.read_calibration_measurements(base + '\\divot_s2_jointangles.m' )[:, 6:13]
#s3 = Yomiread.read_calibration_measurements(base + '\\divot_s3_jointangles.m')[:, 6:13]

result = opt_full_bar_ball_data_check(sb, sb_bb, mb, mb_bb, lb, lb_bb)
print('result is', result)

# np.asarray(ball), mean, variance, stdev, bar_length_diff

time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(base + "\\result"):
    os.makedirs(base + "\\result")
filename1 = base + "\\result\\ball_bar_data_check.csv"
string = 'mean, variance, stdev, bar_length_diff'
#Yomiwrite.write_csv_string(filename1, string)
Yomiwrite.write_csv_matrix(filename1, result)
print('done')





