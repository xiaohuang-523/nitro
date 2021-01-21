import numpy as np
import Kinematics as Yomikin

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
        ee = np.matmul(pos[-1], [0, 0, 0, 1])
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


# Calculate the differences bewteen two lists of vectors (n by 3): vec1, vec2
# ve is the error vector: ve = v1 - v2
# vd is the difference of the length of two vectors vd = length(v1)-length(v2)
# va is the angle between two vectors: va = angle between v1 and v2
def compV_m(vec1, vec2):
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


# Calculate the differences bewteen two vectors (1x3 or 3x1): vec1, vec2
# ve is the error vector: ve = v1 - v2
# vd is the difference of the length of two vectors vd = length(v1)-length(v2)
# va is the angle between two vectors: va = angle between v1 and v2
def compV_s(vec1, vec2):
    vec1_tem = np.copy(vec1)
    vec2_tem = np.copy(vec2)

    ve_tem = vec1_tem - vec2
    ve = np.sqrt(np.sum(ve_tem[0:3] ** 2)) * 1000
    vd = np.sqrt(np.sum(vec1_tem ** 2)) * 1000 - np.sqrt(np.sum(vec2_tem ** 2)) * 1000
    va = np.arccos(np.matmul(vec1_tem, vec2_tem) / (np.sqrt(np.sum(vec1_tem ** 2)) * np.sqrt(np.sum(vec2_tem ** 2))))
    return ve, vd, va


# Compare two dfc nominals
def comp_dfc(dfc1, dfc2):
    dfc1 = np.reshape(dfc1, (4, 4))
    dfc2 = np.reshape(dfc2, (4, 4))
    p1 = dfc1[0:3,3]
    p2 = dfc2[0:3,3]
    x1_vec = dfc1[0:3, 0]
    x2_vec = dfc2[0:3, 0]
    p_error = np.linalg.norm(p1-p2)*1000
    a_error = compV_s(x1_vec, x2_vec)[2]*180/np.pi
    return p_error, a_error


# Compare two calibrations, tracker or ga
def comp_kdl(kdl1, kdl2, pose_list):
    ee1, x2l, x3l, x4l = eeVector(kdl1, pose_list)
    ee2, x2r, x3r, x4r = eeVector(kdl2, pose_list)

    # compare the ee vs. common base
    ve_ms13, vd_ms13, va_ms13 = compV_m(ee1, ee2)
    # compare the ee vs. Joint 1
    #ve_ms13, vd_ms13, va_ms13 = difVector(x2l, x2r)

    ve_mean = np.mean(ve_ms13)
    ve_std = np.std(ve_ms13)
    vd_mean = np.mean(vd_ms13)
    vd_std = np.std(vd_ms13)
    va_mean = np.mean(va_ms13)
    va_std = np.std(va_ms13)

    #print('ve mean and std are', ve_mean, ve_std)
    #print('va mean and std are', va_mean, va_std)
    #print('vd mean and std are', vd_mean, vd_std)
    return ve_mean, ve_std, va_mean*180/np.pi, va_std*180/np.pi


# Compare two transformation matrix (4 by 4)
def comp_mtx(mtx1, mtx2):
    mtx1 = Yomikin.Yomi_Base_Matrix(mtx1)
    mtx2 = Yomikin.Yomi_Base_Matrix(mtx2)
    mtx1 = np.reshape(mtx1, (4, 4))
    mtx2 = np.reshape(mtx2, (4, 4))
    p1 = mtx1[0:3,3]
    p2 = mtx2[0:3,3]
    x1_vec = mtx1[0:3, 0]
    x2_vec = mtx2[0:3, 0]
    p_error = np.linalg.norm(p1-p2)*1000
    a_error_x = compV_s(x1_vec, x2_vec)[2]*180/np.pi
    a_error_y = compV_s(mtx1[0:3, 1], mtx2[0:3, 1])[2]*180/np.pi
    a_error_z = compV_s(mtx1[0:3, 2], mtx2[0:3, 2])[2]*180/np.pi
    return p_error, a_error_x, a_error_y, a_error_z


# Compare the rotation part of the two transformation matrix
# Get the misorientation of two rotation matrices
def comp_mtx_misori(mtx1, mtx2):
    mtx1 = Yomikin.Yomi_Base_Matrix(mtx1)
    mtx2 = Yomikin.Yomi_Base_Matrix(mtx2)
    mtx1 = np.reshape(mtx1, (4, 4))
    mtx2 = np.reshape(mtx2, (4, 4))
    rot1 = mtx1[0:3,0:3]
    rot2 = mtx2[0:3,0:3]
    # the difference between two rotation matrices can be computed as rot1 * (rot2)^(-1)
    # reference: https://en.wikipedia.org/wiki/Misorientation
    delta_rot = np.matmul(rot1, np.linalg.inv(rot2))
    return delta_rot


